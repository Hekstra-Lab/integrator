"""Visualize raw counts vs model predicted rates for the refinement model.

Shows a grid of reflections: observed pixel counts, predicted rate
(s/lp * |F_calc|² * profile + bg), and residuals.

Usage
-----
    uv run python scripts/visualize_refinement_reflections.py \
        --config configs/variational_refinement_hewl.yaml \
        --checkpoint path/to/epoch.ckpt \
        --n 10 --out reflections.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reflections.png")
    parser.add_argument(
        "--sort-by-intensity", action="store_true",
        help="Sort by F_calc (bright to weak) instead of random",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    model = construct_integrator(config)
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    dm = construct_data_loader(config)
    dm.setup("predict")

    # Grab a batch
    dl = dm.predict_dataloader()
    batch = next(iter(dl))
    counts, shoebox, mask, metadata = batch

    # Forward pass
    with torch.no_grad():
        outputs = model(counts, shoebox, mask, metadata)

    fwd = outputs["forward_out"]
    rates = fwd["rates"]
    profiles = fwd["profile"]
    qi_mean = fwd["qi_mean"]
    qbg_mean = fwd["qbg_mean"]

    # rates shape: (B, mc_samples, n_pixels) - take mean over MC samples
    if rates.dim() == 3:
        rates_mean = rates.mean(dim=1)
    else:
        rates_mean = rates

    D = config["integrator"]["args"]["d"]
    H = config["integrator"]["args"]["h"]
    W = config["integrator"]["args"]["w"]
    shape_3d = (D, H, W)

    counts_np = (counts * mask).cpu().numpy()
    rates_np = rates_mean.cpu().numpy()
    mask_np = mask.cpu().numpy()
    profiles_np = profiles.cpu().numpy()
    qi_np = qi_mean.cpu().numpy()
    bg_np = qbg_mean.cpu().numpy()

    n_obs = counts_np.shape[0]

    # Select reflections
    if args.sort_by_intensity:
        order = np.argsort(-qi_np)
        # Pick evenly spaced from bright to weak
        step = max(1, len(order) // args.n)
        idx = order[::step][:args.n]
    else:
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(n_obs, size=min(args.n, n_obs), replace=False)
        idx.sort()

    n = len(idx)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    # For 3D: show middle slice. For 2D: show the single slice.
    mid_z = D // 2

    fig, axes = plt.subplots(
        nrows * 3, ncols, figsize=(2.8 * ncols, 2.5 * nrows * 3),
        squeeze=False,
    )

    for panel_i, obs_i in enumerate(idx):
        col = panel_i % ncols
        row_base = (panel_i // ncols) * 3

        obs_3d = counts_np[obs_i].reshape(shape_3d)
        rate_3d = rates_np[obs_i].reshape(shape_3d)
        prf_3d = profiles_np[obs_i].reshape(shape_3d)

        obs_slice = obs_3d[mid_z]
        rate_slice = rate_3d[mid_z]
        resid_slice = obs_slice - rate_slice

        vmax = max(obs_slice.max(), rate_slice.max(), 1)
        rmax = max(abs(resid_slice.min()), abs(resid_slice.max()), 0.1)

        # Row 1: observed counts
        ax = axes[row_base, col]
        im = ax.imshow(obs_slice, cmap="viridis", origin="lower", vmin=0, vmax=vmax)
        hkl_str = ""
        if "H" in metadata:
            h = int(metadata["H"][obs_i])
            k = int(metadata["K"][obs_i])
            l = int(metadata["L"][obs_i])
            hkl_str = f"({h},{k},{l}) "
        ax.set_title(
            f"{hkl_str}F²={qi_np[obs_i]:.0f}  bg={bg_np[obs_i]:.1f}",
            fontsize=7,
        )
        if col == 0:
            ax.set_ylabel("Observed", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 2: predicted rate
        ax = axes[row_base + 1, col]
        ax.imshow(rate_slice, cmap="viridis", origin="lower", vmin=0, vmax=vmax)
        total_obs = obs_slice.sum()
        total_pred = rate_slice.sum()
        ax.set_title(f"sigmaobs={total_obs:.0f}  sigmapred={total_pred:.0f}", fontsize=7)
        if col == 0:
            ax.set_ylabel("Predicted", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        # Row 3: residual
        ax = axes[row_base + 2, col]
        ax.imshow(resid_slice, cmap="RdBu_r", origin="lower", vmin=-rmax, vmax=rmax)
        ax.set_title(f"RMSD={np.sqrt((resid_slice**2).mean()):.1f}", fontsize=7)
        if col == 0:
            ax.set_ylabel("Residual", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused
    for panel_i in range(n, nrows * ncols):
        col = panel_i % ncols
        row_base = (panel_i // ncols) * 3
        for j in range(3):
            axes[row_base + j, col].set_visible(False)

    fig.suptitle("Observed counts vs model predicted rate (middle z-slice)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
