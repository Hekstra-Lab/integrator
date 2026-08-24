"""Extract refined PDB, merged MTZ, and R-factors from a trained refinement checkpoint.

Produces the crystallographic outputs that replace phenix.refine:
  1. Refined PDB with updated coordinates and B-factors
  2. F_calc (amplitude + phase) from the atomic model
  3. F_obs estimated via profile-fitting from raw pixel data
  4. R-factors (Rwork / Rfree)
  5. Merged MTZ with columns: FP, SIGFP, FC, PHIC, R-free-flags

Map coefficients (2mFo-DFc, mFo-DFc) can be computed from the output MTZ
using phenix.maps, gemmi, or cctbx.

Usage
-----
    uv run python scripts/extract_refinement_outputs.py \
        --config configs/refinement_hewl.yaml \
        --checkpoint path/to/checkpoint.ckpt \
        --out-dir outputs/
"""

import argparse
import logging
from pathlib import Path

import gemmi
import numpy as np
import torch
import yaml

if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )

import reciprocalspaceship as rs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Refined PDB
# ---------------------------------------------------------------------------


def extract_pdb(state_dict: dict, config: dict, out_path: Path) -> None:
    """Write a PDB with refined coordinates and B-factors."""
    import math
    import torch.nn.functional as F

    EIGHT_PI_SQ = 8.0 * math.pi**2

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    pdb_path = int_args["pdb_path"]
    if not Path(pdb_path).is_absolute():
        pdb_path = str(data_dir / pdb_path)

    if "atom_pos_mu" in state_dict:
        atom_pos = state_dict["atom_pos_mu"].cpu().numpy()
        raw_log_sigma = state_dict["atom_raw_log_sigma"].cpu()
        sigma = F.softplus(raw_log_sigma) + 1e-6
        atom_b_iso = (EIGHT_PI_SQ * sigma.pow(2)).numpy()
    else:
        atom_pos = state_dict["atom_pos"].cpu().numpy()
        atom_b_iso = state_dict["atom_b_iso"].cpu().numpy()

    structure = gemmi.read_structure(pdb_path)
    idx = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if idx < len(atom_pos):
                        atom.pos = gemmi.Position(*atom_pos[idx])
                        atom.b_iso = float(atom_b_iso[idx])
                        idx += 1

    structure.write_pdb(str(out_path))
    logger.info("Wrote refined PDB: %s (%d atoms updated)", out_path, idx)


# ---------------------------------------------------------------------------
# 2. F_calc from atomic model
# ---------------------------------------------------------------------------


def compute_f_calc(
    state_dict: dict, config: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute F_calc amplitudes and phases for all ASU HKLs.

    Returns (Hasu_array, F_amp, F_phase_deg, F_sq).
    """
    from SFC_Torch import SFcalculator as SFC

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    pdb_path = int_args["pdb_path"]
    if not Path(pdb_path).is_absolute():
        pdb_path = str(data_dir / pdb_path)

    sfc = SFC(
        pdbmodel=pdb_path,
        dmin=int_args["dmin"],
        anomalous=int_args.get("anomalous", False),
        wavelength=int_args.get("wavelength", 1.0),
        device="cpu",
    )
    sfc.inspect_data()

    if "atom_pos_mu" in state_dict:
        import math
        import torch.nn.functional as F

        EIGHT_PI_SQ = 8.0 * math.pi**2
        sfc.atom_pos_orth = state_dict["atom_pos_mu"].cpu()
        raw_log_sigma = state_dict["atom_raw_log_sigma"].cpu()
        sigma = F.softplus(raw_log_sigma) + 1e-6
        sfc.atom_b_iso = EIGHT_PI_SQ * sigma.pow(2)
    else:
        sfc.atom_pos_orth = state_dict["atom_pos"].cpu()
        sfc.atom_b_iso = state_dict["atom_b_iso"].cpu()

    F_complex = sfc.calc_fprotein(Return=True)

    F_amp = F_complex.abs().detach().numpy()
    F_phase = torch.angle(F_complex).detach().numpy()
    F_phase_deg = np.degrees(F_phase)
    F_sq = (F_complex * F_complex.conj()).real.detach().numpy()
    Hasu = sfc.Hasu_array

    logger.info("Computed F_calc for %d ASU HKLs", len(F_amp))
    return Hasu, F_amp, F_phase_deg, F_sq


# ---------------------------------------------------------------------------
# 3. F_obs via profile-fitting predict pass
# ---------------------------------------------------------------------------


def profile_fit_intensities(
    config: dict,
    checkpoint_path: Path,
) -> dict[str, np.ndarray]:
    """Run predict over the full dataset, profile-fit each observation.

    For each reflection observation:
        I_obs = sum(p * (c - b)) / sum(p^2)
        sigma_I = sqrt(sum(p^2 * c)) / sum(p^2)
        F_sq_obs = I_obs * lp / s(frame)

    Returns dict with per-observation arrays:
        asu_id, H, K, L, I_obs, sigma_I, F_sq_obs, is_test, d
    """
    import pytorch_lightning as pl

    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    model = construct_integrator(config)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
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
        "counts",
        "mask",
        "profile",
        "qbg_mean",
        "asu_id",
        "H",
        "K",
        "L",
        "is_test",
        "d",
        "xyzcal.px.0",
        "xyzcal.px.1",
        "xyzcal.px.2",
        "lp",
    ]

    raw_preds = trainer.predict(model, dataloaders=predict_dl)

    all_asu_id = []
    all_H = []
    all_K = []
    all_L = []
    all_I_obs = []
    all_sigma_I = []
    all_F_sq_obs = []
    all_is_test = []
    all_d = []

    for batch_pred in raw_preds:
        counts = batch_pred["counts"].cpu().float()
        mask = batch_pred["mask"].cpu().float()
        profile = batch_pred["profile"].cpu().float()
        bg_mean = batch_pred["qbg_mean"].cpu().float()
        asu_id = batch_pred["asu_id"].cpu().long()
        lp = batch_pred["lp"].cpu().float().clamp(min=1e-8)
        frame = batch_pred["xyzcal.px.2"].cpu().float()
        is_test = batch_pred.get("is_test", torch.zeros(counts.shape[0])).cpu()

        b = counts.shape[0]
        counts_flat = counts.view(b, -1) * mask.view(b, -1)
        profile_flat = profile.view(b, -1) * mask.view(b, -1)

        signal = counts_flat - bg_mean.view(b, 1)
        p_sq_sum = (profile_flat**2).sum(dim=1).clamp(min=1e-12)

        I_obs = (profile_flat * signal).sum(dim=1) / p_sq_sum
        sigma_I = (profile_flat**2 * counts_flat.clamp(min=0)).sum(dim=1).sqrt() / p_sq_sum

        with torch.no_grad():
            from integrator.model.scaling.chebyshev_scale import (
                SpatialChebyshevScale,
            )

            if isinstance(model.scale_fn, SpatialChebyshevScale):
                x_det = batch_pred["xyzcal.px.0"].cpu().float()
                y_det = batch_pred["xyzcal.px.1"].cpu().float()
                s = model.scale_fn(frame, x_det, y_det)
            else:
                s = model.scale_fn(frame)

        F_sq_obs = I_obs * lp / s.cpu()

        all_asu_id.append(asu_id.numpy())
        all_H.append(batch_pred["H"].cpu().numpy())
        all_K.append(batch_pred["K"].cpu().numpy())
        all_L.append(batch_pred["L"].cpu().numpy())
        all_I_obs.append(I_obs.numpy())
        all_sigma_I.append(sigma_I.numpy())
        all_F_sq_obs.append(F_sq_obs.numpy())
        all_is_test.append(is_test.numpy())
        all_d.append(batch_pred["d"].cpu().numpy())

    return {
        "asu_id": np.concatenate(all_asu_id),
        "H": np.concatenate(all_H),
        "K": np.concatenate(all_K),
        "L": np.concatenate(all_L),
        "I_obs": np.concatenate(all_I_obs),
        "sigma_I": np.concatenate(all_sigma_I),
        "F_sq_obs": np.concatenate(all_F_sq_obs),
        "is_test": np.concatenate(all_is_test),
        "d": np.concatenate(all_d),
    }


# ---------------------------------------------------------------------------
# 4. Merge observations and compute R-factors
# ---------------------------------------------------------------------------


def merge_observations(
    obs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Inverse-variance weighted merge of per-observation F_sq_obs by asu_id.

    Returns per-unique-HKL arrays:
        asu_id, H, K, L, F_sq_obs, sigma_F_sq, n_obs, is_test (majority vote)
    """
    asu_ids = obs["asu_id"]
    n_unique = int(asu_ids.max()) + 1

    F_sq_sum_w = np.zeros(n_unique, dtype=np.float64)
    w_sum = np.zeros(n_unique, dtype=np.float64)
    n_obs = np.zeros(n_unique, dtype=np.int32)
    n_test = np.zeros(n_unique, dtype=np.int32)
    H = np.zeros(n_unique, dtype=np.int32)
    K = np.zeros(n_unique, dtype=np.int32)
    L = np.zeros(n_unique, dtype=np.int32)

    sigma_sq = np.clip(obs["sigma_I"] ** 2, 1e-12, None)
    w = 1.0 / sigma_sq

    for i in range(len(asu_ids)):
        aid = asu_ids[i]
        F_sq_sum_w[aid] += w[i] * obs["F_sq_obs"][i]
        w_sum[aid] += w[i]
        n_obs[aid] += 1
        if obs["is_test"][i]:
            n_test[aid] += 1
        if n_obs[aid] == 1:
            H[aid] = int(obs["H"][i])
            K[aid] = int(obs["K"][i])
            L[aid] = int(obs["L"][i])

    observed = w_sum > 0
    F_sq_merged = np.where(observed, F_sq_sum_w / np.clip(w_sum, 1e-12, None), 0.0)
    sigma_merged = np.where(observed, 1.0 / np.sqrt(np.clip(w_sum, 1e-12, None)), 0.0)
    is_test = n_test > (n_obs / 2)

    return {
        "asu_id": np.arange(n_unique),
        "H": H,
        "K": K,
        "L": L,
        "F_sq_obs": F_sq_merged,
        "sigma_F_sq": sigma_merged,
        "n_obs": n_obs,
        "is_test": is_test,
        "observed": observed,
    }


def compute_r_factors(
    merged: dict[str, np.ndarray],
    F_calc_sq: np.ndarray,
    Hasu: np.ndarray,
    config: dict,
) -> dict[str, float]:
    """Compute Rwork and Rfree on amplitude scale.

    R = sum ||Fo| - k|Fc|| / sum |Fo|

    where k is a global scale factor minimizing R.
    Maps merged asu_ids to SFcalculator's Hasu indices to align F_obs with F_calc.
    """
    from integrator.model.scaling.refinement_integrator import _build_hasu_lookup

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    asu_id_to_hkl_path = int_args["asu_id_to_hkl_path"]
    if not Path(asu_id_to_hkl_path).is_absolute():
        asu_id_to_hkl_path = str(data_dir / asu_id_to_hkl_path)

    id_to_hkl = torch.load(asu_id_to_hkl_path, weights_only=False, map_location="cpu")

    sg = gemmi.SpaceGroup(
        yaml.safe_load(
            (data_dir / "crystal.yaml").read_text()
        ).get("space_group", "P1").split("(")[0].strip()
    )
    hasu_lookup = _build_hasu_lookup(Hasu, sg, int_args.get("anomalous", False))

    obs_mask = merged["observed"]
    F_obs = np.sqrt(np.clip(merged["F_sq_obs"], 0, None))
    is_test = merged["is_test"]

    F_calc_aligned = np.zeros(len(merged["asu_id"]), dtype=np.float64)
    mapped = np.zeros(len(merged["asu_id"]), dtype=bool)

    for aid in range(len(merged["asu_id"])):
        if not obs_mask[aid] or aid >= len(id_to_hkl):
            continue
        hkl = tuple(int(x) for x in id_to_hkl[aid])
        if hkl in hasu_lookup:
            sfc_idx = hasu_lookup[hkl]
            F_calc_aligned[aid] = np.sqrt(max(F_calc_sq[sfc_idx], 0))
            mapped[aid] = True

    valid = obs_mask & mapped & (F_obs > 0)

    work = valid & ~is_test
    test = valid & is_test

    def _r_factor(sel):
        fo = F_obs[sel]
        fc = F_calc_aligned[sel]
        if len(fo) == 0:
            return float("nan")
        k = np.sum(fo * fc) / np.sum(fc * fc)
        return float(np.sum(np.abs(fo - k * fc)) / np.sum(fo))

    r_work = _r_factor(work)
    r_free = _r_factor(test)
    n_work = int(work.sum())
    n_free = int(test.sum())
    n_mapped = int(mapped.sum())

    logger.info(
        "R-factors: Rwork=%.4f (%d refl), Rfree=%.4f (%d refl), %d mapped",
        r_work,
        n_work,
        r_free,
        n_free,
        n_mapped,
    )
    return {
        "r_work": r_work,
        "r_free": r_free,
        "n_work": n_work,
        "n_free": n_free,
    }


# ---------------------------------------------------------------------------
# 5. Write merged MTZ
# ---------------------------------------------------------------------------


def write_mtz(
    merged: dict[str, np.ndarray],
    F_calc_sq: np.ndarray,
    F_phase_deg: np.ndarray,
    Hasu: np.ndarray,
    config: dict,
    out_path: Path,
) -> None:
    """Write merged MTZ with FP, SIGFP, FC, PHIC, FreeR_flag columns."""
    from integrator.model.scaling.refinement_integrator import _build_hasu_lookup

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])
    asu_id_to_hkl_path = int_args["asu_id_to_hkl_path"]
    if not Path(asu_id_to_hkl_path).is_absolute():
        asu_id_to_hkl_path = str(data_dir / asu_id_to_hkl_path)

    id_to_hkl = torch.load(asu_id_to_hkl_path, weights_only=False, map_location="cpu")

    crystal_yaml = data_dir / "crystal.yaml"
    crystal = yaml.safe_load(crystal_yaml.read_text())
    cell = gemmi.UnitCell(*crystal["cell"])
    sg_str = crystal.get("space_group", "P1").split("(")[0].strip()
    spacegroup = gemmi.SpaceGroup(sg_str)

    hasu_lookup = _build_hasu_lookup(Hasu, spacegroup, int_args.get("anomalous", False))

    obs_mask = merged["observed"]
    F_obs = np.sqrt(np.clip(merged["F_sq_obs"], 0, None))
    sig_F_obs = np.where(
        F_obs > 0,
        merged["sigma_F_sq"] / (2 * np.clip(F_obs, 1e-12, None)),
        0.0,
    )

    rows = []
    for aid in range(len(merged["asu_id"])):
        if not obs_mask[aid] or aid >= len(id_to_hkl):
            continue
        hkl = tuple(int(x) for x in id_to_hkl[aid])
        if hkl not in hasu_lookup:
            continue
        sfc_idx = hasu_lookup[hkl]

        rows.append(
            {
                "H": merged["H"][aid],
                "K": merged["K"][aid],
                "L": merged["L"][aid],
                "FP": F_obs[aid],
                "SIGFP": sig_F_obs[aid],
                "FC": np.sqrt(max(F_calc_sq[sfc_idx], 0)),
                "PHIC": F_phase_deg[sfc_idx],
                "FreeR_flag": int(merged["is_test"][aid]),
            }
        )

    import pandas as pd

    df = pd.DataFrame(rows)
    ds = rs.DataSet(df, cell=cell, spacegroup=spacegroup).infer_mtz_dtypes()

    valid = ds["SIGFP"] > 0
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        logger.info("Dropped %d reflections with SIGFP=0", n_dropped)
        ds = ds[valid]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.write_mtz(str(out_path), skip_problem_mtztypes=True)
    logger.info(
        "Wrote %s: %d reflections, cell=%s, sg=%s",
        out_path,
        len(ds),
        cell,
        spacegroup.hm,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract refinement outputs from a trained checkpoint."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("refinement_outputs"))
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip the predict pass (PDB + F_calc only, no F_obs or R-factors).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # 1. Refined PDB
    pdb_path = args.out_dir / "refined.pdb"
    extract_pdb(state_dict, config, pdb_path)

    # 2. F_calc
    Hasu, F_amp, F_phase_deg, F_sq = compute_f_calc(state_dict, config)

    if args.skip_predict:
        logger.info("--skip-predict: stopping after PDB + F_calc.")
        return

    # 3. Profile-fitted F_obs from pixel data
    logger.info("Running predict pass for profile-fitted F_obs...")
    obs = profile_fit_intensities(config, args.checkpoint)

    # Load is_test from metadata if predict didn't provide it
    if obs["is_test"].sum() == 0:
        data_dir = Path(config["data_loader"]["args"]["data_dir"])
        ref_name = (
            config["data_loader"]["args"]
            .get("shoebox_file_names", {})
            .get("reference", "metadata.pt")
        )
        meta = torch.load(data_dir / ref_name, weights_only=False, map_location="cpu")
        if "is_test" in meta:
            obs["is_test"] = meta["is_test"].numpy().astype(bool)
            logger.info("Loaded is_test from metadata: %d test obs", obs["is_test"].sum())

    # 4. Merge and R-factors
    merged = merge_observations(obs)
    r = compute_r_factors(merged, F_sq, Hasu, config)

    summary_path = args.out_dir / "refinement_stats.txt"
    with open(summary_path, "w") as f:
        f.write(f"Rwork:  {r['r_work']:.4f}  ({r['n_work']} reflections)\n")
        f.write(f"Rfree:  {r['r_free']:.4f}  ({r['n_free']} reflections)\n")
    logger.info("Wrote %s", summary_path)

    # 5. Merged MTZ
    mtz_path = args.out_dir / "refined.mtz"
    write_mtz(merged, F_sq, F_phase_deg, Hasu, config, mtz_path)

    logger.info("Done. For map coefficients, run:")
    logger.info("  phenix.maps refined.pdb refined.mtz")
    logger.info("  or: gemmi sf2map refined.mtz -o map.ccp4")


if __name__ == "__main__":
    main()
