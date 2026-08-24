"""Compare refinement model intensities vs DIALS.

Computes I_model = s(frame,r)/lp * |F_calc|^2 for each observation
and plots against DIALS intensity.prf.value.

Usage
-----
    uv run python scripts/compare_refinement_intensities.py \
        --config configs/variational_refinement_hewl.yaml \
        --checkpoint path/to/epoch.ckpt \
        --out intensity_comparison.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

import gemmi

if not hasattr(gemmi.UnitCell, "fractionalization_matrix"):
    gemmi.UnitCell.fractionalization_matrix = property(
        lambda self: self.frac.mat
    )
if not hasattr(gemmi.UnitCell, "orthogonalization_matrix"):
    gemmi.UnitCell.orthogonalization_matrix = property(
        lambda self: self.orth.mat
    )

EIGHT_PI_SQ = 8.0 * math.pi**2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=str, default="intensity_comparison.png")
    parser.add_argument("--max-obs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    int_args = config["integrator"]["args"]
    data_dir = Path(config["data_loader"]["args"]["data_dir"])

    # Load metadata
    ref_name = (
        config["data_loader"]["args"]
        .get("shoebox_file_names", {})
        .get("reference", "metadata.pt")
    )
    meta = torch.load(data_dir / ref_name, weights_only=False, map_location="cpu")

    asu_ids = meta["asu_id"].long()
    lp = meta["lp"].float().clamp(min=1e-8)
    frame = meta["xyzcal.px.2"].float()
    I_dials = meta["intensity.prf.value"].numpy()

    n_obs = len(asu_ids)
    if args.max_obs and args.max_obs < n_obs:
        idx = np.random.choice(n_obs, args.max_obs, replace=False)
        idx.sort()
    else:
        idx = np.arange(n_obs)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    state_dict = ckpt["state_dict"]

    # Compute F_calc^2 from atomic model
    from SFC_Torch import SFcalculator as SFC
    from integrator.model.scaling.refinement_integrator import _build_hasu_lookup

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
        sfc.atom_pos_orth = state_dict["atom_pos_mu"].cpu()
        raw_log_sigma = state_dict["atom_raw_log_sigma"].cpu()
        sigma = F.softplus(raw_log_sigma) + 1e-6
        sfc.atom_b_iso = EIGHT_PI_SQ * sigma.pow(2)
    else:
        sfc.atom_pos_orth = state_dict["atom_pos"].cpu()
        sfc.atom_b_iso = state_dict["atom_b_iso"].cpu()

    Fc = sfc.calc_fprotein(Return=True).detach()

    # Bulk solvent
    if int_args.get("bulk_solvent", False) and "raw_k_sol" in state_dict:
        sfc.calc_fsolvent()
        F_mask = sfc.Fmask_asu.detach()
        d_hkl = torch.from_numpy(sfc.dHasu).float()
        s_sq = 1.0 / (4.0 * d_hkl.pow(2))
        k_sol = F.softplus(state_dict["raw_k_sol"].cpu())
        B_sol = F.softplus(state_dict["raw_B_sol"].cpu())
        dampening = torch.exp(-B_sol * s_sq)
        Fc = Fc + k_sol * dampening * F_mask

    F_sq_all = (Fc * Fc.conj()).real

    # Build asu_id -> SFcalculator index mapping
    asu_id_to_hkl_path = int_args["asu_id_to_hkl_path"]
    if not Path(asu_id_to_hkl_path).is_absolute():
        asu_id_to_hkl_path = str(data_dir / asu_id_to_hkl_path)
    id_to_hkl = torch.load(asu_id_to_hkl_path, weights_only=False, map_location="cpu")

    sg = gemmi.SpaceGroup(
        yaml.safe_load((data_dir / "crystal.yaml").read_text())
        .get("space_group", "P1").split("(")[0].strip()
    )
    hasu_lookup = _build_hasu_lookup(sfc.Hasu_array, sg)

    n_asu_ids = len(id_to_hkl)
    sfc_idx = torch.full((n_asu_ids,), 0, dtype=torch.long)
    for aid in range(n_asu_ids):
        h, k, l = int(id_to_hkl[aid, 0]), int(id_to_hkl[aid, 1]), int(id_to_hkl[aid, 2])
        if (h, k, l) in hasu_lookup:
            sfc_idx[aid] = hasu_lookup[(h, k, l)]

    # Reconstruct scale function
    from integrator.model.scaling.chebyshev_scale import (
        ChebyshevScale,
        SpatialChebyshevScale,
    )

    if int_args.get("scale_spatial", False):
        scale_fn = SpatialChebyshevScale(
            degree_frame=int_args["scale_degree"],
            degree_radius=int_args.get("scale_degree_radius", 5),
            frame_min=int_args["scale_frame_min"],
            frame_max=int_args["scale_frame_max"],
            beam_center=int_args.get("scale_beam_center"),
            r_min=int_args.get("scale_r_min", 0.0),
            r_max=int_args.get("scale_r_max", 1500.0),
        )
    else:
        scale_fn = ChebyshevScale(
            degree=int_args["scale_degree"],
            frame_min=int_args["scale_frame_min"],
            frame_max=int_args["scale_frame_max"],
        )

    # Load scale weights from checkpoint
    scale_state = {
        k.replace("scale_fn.", ""): v
        for k, v in state_dict.items()
        if k.startswith("scale_fn.")
    }
    scale_fn.load_state_dict(scale_state)
    scale_fn.eval()

    # Compute I_model = s/lp * F^2 for selected observations
    with torch.no_grad():
        obs_asu = asu_ids[idx]
        obs_frame = frame[idx]
        obs_lp = lp[idx]

        F_sq = F_sq_all[sfc_idx[obs_asu]]

        if isinstance(scale_fn, SpatialChebyshevScale):
            x_det = meta["xyzcal.px.0"].float()[idx]
            y_det = meta["xyzcal.px.1"].float()[idx]
            s = scale_fn(obs_frame, x_det, y_det)
        else:
            s = scale_fn(obs_frame)

        I_model = (s / obs_lp * F_sq).numpy()

    I_dials_sel = I_dials[idx]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter
    ax = axes[0]
    valid = (I_model > 0) & (I_dials_sel > 0)
    ax.scatter(
        I_model[valid], I_dials_sel[valid],
        s=0.3, alpha=0.02, c="steelblue", edgecolors="none",
    )
    lims = [1, max(np.percentile(I_model[valid], 99.5), np.percentile(I_dials_sel[valid], 99.5))]
    ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7, label="x = y")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Model: s/lp * |F_calc|²")
    ax.set_ylabel("DIALS: intensity.prf.value")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    corr = np.corrcoef(np.log(I_model[valid] + 1), np.log(I_dials_sel[valid] + 1))[0, 1]
    ax.set_title(f"Log-correlation: {corr:.4f}  |  N={valid.sum():,}")

    # Right: ratio histogram
    ax = axes[1]
    ratio = I_model[valid] / I_dials_sel[valid]
    ratio_clipped = ratio[(ratio > 0.01) & (ratio < 100)]
    ax.hist(np.log10(ratio_clipped), bins=100, color="steelblue", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("log10(I_model / I_dials)")
    ax.set_ylabel("Count")
    ax.set_title(f"Median ratio: {np.median(ratio):.3f}")

    plt.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")
    print(f"  Pearson corr (log): {corr:.4f}")
    print(f"  Median I_model/I_dials: {np.median(ratio):.3f}")
    print(f"  N valid: {valid.sum():,} / {len(idx):,}")


if __name__ == "__main__":
    main()
