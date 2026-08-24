"""Compute observed anomalous difference map from a refinement model.

Extracts F_obs via profile-fitting from raw pixel data using the model's
learned profiles, backgrounds, and scale. Applies French-Wilson via
rs.algorithms.scale_merged_intensities to handle negative/weak intensities.
Phases come from the model's F_calc.

Caches intermediate results in the output directory so that expensive
steps (predict pass) are not repeated on re-runs.

Usage
-----
    uv run python scripts/anomalous_map.py \
        --config configs/variational_refinement_hewl.yaml \
        --checkpoint path/to/epoch.ckpt \
        --out test/anomalous.mtz
"""

import logging
import math
from pathlib import Path

import gemmi
import numpy as np
import torch
import torch.nn.functional as Fn
import yaml

import reciprocalspaceship as rs
from reciprocalspaceship.algorithms import scale_merged_intensities

if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EIGHT_PI_SQ = 8.0 * math.pi**2


# ------------------------------------------------------------------
# Step 1: predict pass + profile fitting + merge
# ------------------------------------------------------------------


def profile_fit_and_merge(config, checkpoint_path, cache_dir):
    cache_file = cache_dir / "profile_fitted_merged.npz"
    if cache_file.exists():
        logger.info("Loading cached profile-fitted data from %s", cache_file)
        data = np.load(cache_file)
        return data["hkl"], data["I_merged"], data["sig_I_merged"]

    import pytorch_lightning as pl

    from integrator.model.scaling.chebyshev_scale import SpatialChebyshevScale
    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    model = construct_integrator(config)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    model_state = model.state_dict()
    compat_state = {
        k: v for k, v in ckpt["state_dict"].items()
        if k not in model_state or v.shape == model_state[k].shape
    }
    model.load_state_dict(compat_state, strict=False)
    model.eval()

    dm = construct_data_loader(config)
    dm.setup("predict")
    predict_dl = dm.predict_dataloader()

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    model.predict_keys = [
        "counts", "mask", "profile", "qbg_mean",
        "H", "K", "L", "d",
        "xyzcal.px.0", "xyzcal.px.1", "xyzcal.px.2", "lp",
    ]

    logger.info("Running predict pass...")
    raw_preds = trainer.predict(model, dataloaders=predict_dl)

    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    sg_str = yaml.safe_load((data_dir / "crystal.yaml").read_text()).get(
        "space_group", "P1"
    ).split("(")[0].strip()
    sg = gemmi.SpaceGroup(sg_str)

    all_F_sq = []
    all_sig_F_sq = []
    all_H = []
    all_K = []
    all_L = []

    for batch_pred in raw_preds:
        counts = batch_pred["counts"].cpu().float()
        mask = batch_pred["mask"].cpu().float()
        profile = batch_pred["profile"].cpu().float()
        bg_mean = batch_pred["qbg_mean"].cpu().float()
        lp = batch_pred["lp"].cpu().float().clamp(min=1e-8)
        frame = batch_pred["xyzcal.px.2"].cpu().float()

        b = counts.shape[0]
        counts_flat = counts.view(b, -1) * mask.view(b, -1)
        profile_flat = profile.view(b, -1) * mask.view(b, -1)

        # Kabsch/Otwinowski profile fitting
        signal = counts_flat - bg_mean.view(b, 1)
        bg_per_pixel = bg_mean.view(b, 1).expand_as(counts_flat)

        v = bg_per_pixel.clamp(min=0.1)
        I_obs = (signal * profile_flat / v).sum(dim=1) / (profile_flat**2 / v).sum(dim=1).clamp(min=1e-12)

        for _ in range(3):
            v = (bg_per_pixel + I_obs.clamp(min=0).unsqueeze(1) * profile_flat).clamp(min=0.1)
            I_new = (signal * profile_flat / v).sum(dim=1) / (profile_flat**2 / v).sum(dim=1).clamp(min=1e-12)
            converged = (I_new - I_obs).abs() < 0.01
            I_obs = torch.where(converged, I_obs, I_new)

        sigma_I = 1.0 / (profile_flat**2 / v).sum(dim=1).clamp(min=1e-12).sqrt()

        with torch.no_grad():
            if isinstance(model.scale_fn, SpatialChebyshevScale):
                x_det = batch_pred["xyzcal.px.0"].cpu().float()
                y_det = batch_pred["xyzcal.px.1"].cpu().float()
                s = model.scale_fn(frame, x_det, y_det)
            else:
                s = model.scale_fn(frame)

        all_F_sq.append((I_obs * lp / s.cpu()).numpy())
        all_sig_F_sq.append((sigma_I * lp / s.cpu()).numpy())
        all_H.append(batch_pred["H"].cpu().numpy())
        all_K.append(batch_pred["K"].cpu().numpy())
        all_L.append(batch_pred["L"].cpu().numpy())

    all_F_sq = np.concatenate(all_F_sq)
    all_sig_F_sq = np.concatenate(all_sig_F_sq)
    all_H = np.concatenate(all_H).astype(np.int32)
    all_K = np.concatenate(all_K).astype(np.int32)
    all_L = np.concatenate(all_L).astype(np.int32)

    # Map raw HKLs to anomalous ASU
    hkl_obs = np.stack([all_H, all_K, all_L], axis=1)
    asu_hkl, isym = rs.utils.hkl_to_asu(hkl_obs, sg)
    is_minus = (isym % 2 == 0)
    canon_hkl = asu_hkl.copy()
    canon_hkl[is_minus] = -canon_hkl[is_minus]

    # Merge by canonical HKL
    canon_to_id: dict[tuple[int, int, int], int] = {}
    for i in range(len(canon_hkl)):
        key = (int(canon_hkl[i, 0]), int(canon_hkl[i, 1]), int(canon_hkl[i, 2]))
        if key not in canon_to_id:
            canon_to_id[key] = len(canon_to_id)

    n_unique = len(canon_to_id)
    I_sum_w = np.zeros(n_unique, dtype=np.float64)
    w_sum = np.zeros(n_unique, dtype=np.float64)

    for i in range(len(all_F_sq)):
        key = (int(canon_hkl[i, 0]), int(canon_hkl[i, 1]), int(canon_hkl[i, 2]))
        aid = canon_to_id[key]
        v = all_sig_F_sq[i] ** 2
        if v > 0 and np.isfinite(all_F_sq[i]):
            w = 1.0 / v
            I_sum_w[aid] += w * all_F_sq[i]
            w_sum[aid] += w

    I_merged = np.where(w_sum > 0, I_sum_w / w_sum, np.nan)
    sig_I_merged = np.where(w_sum > 0, 1.0 / np.sqrt(w_sum), np.nan)

    hkl = np.zeros((n_unique, 3), dtype=np.int32)
    for key, aid in canon_to_id.items():
        hkl[aid] = key

    observed = np.isfinite(I_merged)
    n_neg = (I_merged[observed] < 0).sum()
    logger.info(
        "Profile-fitted and merged: %d / %d unique HKLs observed, "
        "%d negative I_merged (%.1f%%)",
        observed.sum(), n_unique,
        n_neg, 100 * n_neg / max(observed.sum(), 1),
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, hkl=hkl, I_merged=I_merged, sig_I_merged=sig_I_merged)
    logger.info("Cached profile-fitted data to %s", cache_file)

    return hkl, I_merged, sig_I_merged


# ------------------------------------------------------------------
# Step 2: phases from refined model
# ------------------------------------------------------------------


def compute_phases(config, checkpoint_path, cache_dir):
    cache_file = cache_dir / "phases.npz"
    if cache_file.exists():
        logger.info("Loading cached phases from %s", cache_file)
        data = np.load(cache_file, allow_pickle=True)
        return data["phase_hkls"], data["phase_vals"]

    from SFC_Torch import SFcalculator as SFC

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    pdb_path = int_args["pdb_path"]
    if not Path(pdb_path).is_absolute():
        pdb_path = str(data_dir / pdb_path)

    sfc = SFC(
        pdbmodel=pdb_path,
        dmin=int_args["dmin"],
        anomalous=True,
        wavelength=int_args.get("wavelength", 1.0),
        device="cpu",
    )
    sfc.inspect_data()

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]

    if "atom_pos_mu" in state_dict:
        sfc.atom_pos_orth = state_dict["atom_pos_mu"].cpu()
        raw_log_sigma = state_dict["atom_raw_log_sigma"].cpu()
        sigma = Fn.softplus(raw_log_sigma) + 1e-6
        sfc.atom_b_iso = EIGHT_PI_SQ * sigma.pow(2)
    else:
        sfc.atom_pos_orth = state_dict["atom_pos"].cpu()
        sfc.atom_b_iso = state_dict["atom_b_iso"].cpu()

    Fc = sfc.calc_fprotein(Return=True).detach()
    phase_deg = np.degrees(torch.angle(Fc).numpy())

    sg = sfc.space_group
    op_list = list(sg.operations())

    # Build lookup arrays instead of dict (for caching)
    phase_hkls = []
    phase_vals = []
    for idx in range(len(sfc.Hasu_array)):
        h, k, l = int(sfc.Hasu_array[idx, 0]), int(sfc.Hasu_array[idx, 1]), int(sfc.Hasu_array[idx, 2])
        for op in op_list:
            hkl_rot = op.apply_to_hkl([h, k, l])
            phase_hkls.append(tuple(hkl_rot))
            phase_vals.append(phase_deg[idx])
            phase_hkls.append((-hkl_rot[0], -hkl_rot[1], -hkl_rot[2]))
            phase_vals.append(-phase_deg[idx])

    phase_hkls = np.array(phase_hkls, dtype=np.int32)
    phase_vals = np.array(phase_vals, dtype=np.float64)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, phase_hkls=phase_hkls, phase_vals=phase_vals)
    logger.info("Cached phases to %s (%d HKLs)", cache_file, len(sfc.Hasu_array))

    return phase_hkls, phase_vals


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("anomalous.mtz"))
    parser.add_argument("--recompute", action="store_true", help="Ignore cached files")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    crystal = yaml.safe_load((data_dir / "crystal.yaml").read_text())
    cell = gemmi.UnitCell(*crystal["cell"])
    sg_str = crystal.get("space_group", "P1").split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    cache_dir = args.out.parent
    if args.recompute:
        for f in cache_dir.glob("profile_fitted_merged.npz"):
            f.unlink()
        for f in cache_dir.glob("phases.npz"):
            f.unlink()

    # 1. Profile-fit and merge observed intensities
    logger.info("Step 1: Profile-fitting from pixel data...")
    hkl, I_merged, sig_I_merged = profile_fit_and_merge(
        config, args.checkpoint, cache_dir
    )

    # 2. Build DataSet with merged I_obs
    observed = np.isfinite(I_merged)
    ds = rs.DataSet(
        {
            "H": rs.DataSeries(hkl[observed, 0], dtype="H"),
            "K": rs.DataSeries(hkl[observed, 1], dtype="H"),
            "L": rs.DataSeries(hkl[observed, 2], dtype="H"),
            "I": rs.DataSeries(I_merged[observed].astype(np.float64), dtype="J"),
            "SIGI": rs.DataSeries(sig_I_merged[observed].astype(np.float64), dtype="Q"),
        },
        cell=cell,
        spacegroup=spacegroup,
        merged=True,
    )
    ds = ds.set_index(["H", "K", "L"])

    logger.info("Step 2: Applying French-Wilson to %d reflections...", len(ds))

    # 3. French-Wilson
    ds = scale_merged_intensities(
        ds, intensity_key="I", sigma_key="SIGI",
        output_columns=["FW-I", "FW-SIGI", "FW-F", "FW-SIGF"],
    )
    logger.info(
        "French-Wilson: F range [%.2f, %.2f], %d reflections",
        ds["FW-F"].min(), ds["FW-F"].max(), len(ds),
    )

    # 4. Unstack anomalous
    ds_anom = ds[["FW-F", "FW-SIGF"]].copy()
    ds_anom = ds_anom.rename(columns={"FW-F": "F", "FW-SIGF": "SIGF"})
    ds_anom["F"] = ds_anom["F"].astype("SFAmplitude")
    ds_anom["SIGF"] = ds_anom["SIGF"].astype("Stddev")
    ds_anom = ds_anom.unstack_anomalous()

    logger.info(
        "Unstacked anomalous: %d reflections with F(+)/F(-)", len(ds_anom),
    )

    # 5. Compute DANO and add phases
    ds_anom["ANOM"] = ds_anom["F(+)"] - ds_anom["F(-)"]

    logger.info("Step 3: Computing phases from refined model...")
    phase_hkls, phase_vals = compute_phases(config, args.checkpoint, cache_dir)

    # Rebuild phase lookup dict from cached arrays
    phase_lookup: dict[tuple[int, int, int], float] = {}
    for i in range(len(phase_hkls)):
        phase_lookup[tuple(phase_hkls[i])] = float(phase_vals[i])

    hkls = ds_anom.get_hkls()
    panom = np.zeros(len(ds_anom), dtype=np.float64)
    n_phased = 0
    for i, (h, k, l) in enumerate(hkls):
        key = (int(h), int(k), int(l))
        if key in phase_lookup:
            panom[i] = phase_lookup[key] - 90.0
            n_phased += 1
    ds_anom["PANOM"] = rs.DataSeries(panom, index=ds_anom.index, dtype="Phase")

    # Filter to complete Friedel pairs
    valid = ds_anom["F(+)"].notna() & ds_anom["F(-)"].notna()
    ds_out = ds_anom[valid].copy()
    ds_out["ANOM"] = ds_out["ANOM"].astype("SFAmplitude")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds_out.write_mtz(str(args.out), skip_problem_mtztypes=True)

    dano = ds_out["ANOM"].to_numpy(dtype=np.float64)
    logger.info("Wrote %s: %d reflections", args.out, len(ds_out))
    logger.info("  %d reflections phased", n_phased)
    logger.info(
        "  DANO: mean=%.3f, std=%.3f, |max|=%.3f",
        np.nanmean(dano), np.nanstd(dano), np.nanmax(np.abs(dano)),
    )
    logger.info("")
    logger.info("Find peaks:")
    logger.info(
        "  rs.find_peaks %s <refined.pdb> -f ANOM -p PANOM -z 5.0 -o peaks.csv",
        args.out,
    )


if __name__ == "__main__":
    main()
