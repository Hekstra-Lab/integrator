"""Visualize the position-aware profile bias at different detector positions.

Shows how the anisotropic Gaussian bias orients radially for different
positions on the detector, plus the fixed decoder bias and their sum.

Usage:
    uv run python scripts/visualize_posaware_bias.py <checkpoint.ckpt> [--out bias_grid.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize position-aware profile bias"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    epoch = ckpt.get("epoch", "?")

    # Check for position-aware model
    if "surrogates.qp.raw_sigma_radial" not in sd:
        print("Not a position-aware profile model (no raw_sigma_radial found)")
        return

    # Extract parameters
    raw_sr = sd["surrogates.qp.raw_sigma_radial"]  # [a, b]
    raw_st = sd["surrogates.qp.raw_sigma_tangential"]  # [a, b]
    beam_cx = sd["surrogates.qp.beam_cx"].item()
    beam_cy = sd["surrogates.qp.beam_cy"].item()
    pixel_x = sd["surrogates.qp.pixel_x"]  # (K,)
    pixel_y = sd["surrogates.qp.pixel_y"]  # (K,)
    decoder_bias = sd["surrogates.qp.decoder.bias"]  # (K,)

    K = pixel_x.shape[0]
    H = W = int(np.sqrt(K))

    print(f"Epoch: {epoch}")
    print(f"Beam center: ({beam_cx:.0f}, {beam_cy:.0f})")
    print(f"Shoebox: {H}x{W}")
    print(f"raw_sigma_radial: {raw_sr.tolist()}")
    print(f"raw_sigma_tangential: {raw_st.tolist()}")

    # Sample detector positions: grid around beam center
    positions = [
        ("center", beam_cx, beam_cy),
        ("right", beam_cx + 800, beam_cy),
        ("left", beam_cx - 800, beam_cy),
        ("top", beam_cx, beam_cy + 800),
        ("bottom", beam_cx, beam_cy - 800),
        ("top-right", beam_cx + 600, beam_cy + 600),
        ("top-left", beam_cx - 600, beam_cy + 600),
        ("bot-right", beam_cx + 600, beam_cy - 600),
        ("bot-left", beam_cx - 600, beam_cy - 600),
        ("far right", beam_cx + 1400, beam_cy),
        ("far top", beam_cx, beam_cy + 1400),
        ("far top-right", beam_cx + 1000, beam_cy + 1000),
    ]

    n = len(positions)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    # Plot 1: position bias only (anisotropic Gaussian)
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes1 = [axes1]

    # Plot 2: softmax(position_bias + decoder_bias) - the actual default profile
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes2 = [axes2]

    for i, (label, xcal, ycal) in enumerate(positions):
        r_idx = i // ncols
        c_idx = i % ncols

        dx = xcal - beam_cx
        dy = ycal - beam_cy
        r = max(np.sqrt(dx**2 + dy**2), 1.0)
        ux = dx / r
        uy = dy / r

        sigma_r = F.softplus(raw_sr[0] + raw_sr[1] * r / 1000.0).item() + 0.5
        sigma_t = F.softplus(raw_st[0] + raw_st[1] * r / 1000.0).item() + 0.5

        proj_r = pixel_x * ux + pixel_y * uy
        proj_t = -pixel_x * uy + pixel_y * ux

        log_profile = -0.5 * ((proj_r / sigma_r)**2 + (proj_t / sigma_t)**2)

        # Position bias only
        pos_bias = log_profile.reshape(H, W).numpy()
        ax1 = axes1[r_idx][c_idx]
        im1 = ax1.imshow(pos_bias, cmap="cividis", origin="lower")
        fig1.colorbar(im1, ax=ax1, shrink=0.8)
        ax1.set_title(f"{label}\nr={r:.0f}  sigmar={sigma_r:.2f}  sigmat={sigma_t:.2f}", fontsize=7)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Combined: softmax(position_bias + decoder_bias)
        combined = F.softmax(log_profile + decoder_bias, dim=0).reshape(H, W).numpy()
        ax2 = axes2[r_idx][c_idx]
        im2 = ax2.imshow(combined, cmap="cividis", origin="lower")
        fig2.colorbar(im2, ax=ax2, shrink=0.8)
        ax2.set_title(f"{label}\nr={r:.0f}", fontsize=7)
        ax2.set_xticks([])
        ax2.set_yticks([])

    # Hide unused
    for i in range(n, nrows * ncols):
        r_idx = i // ncols
        c_idx = i % ncols
        axes1[r_idx][c_idx].set_visible(False)
        axes2[r_idx][c_idx].set_visible(False)

    fig1.suptitle(f"Position bias (anisotropic Gaussian) - epoch {epoch}", fontsize=11)
    fig1.tight_layout()

    fig2.suptitle(f"Default profile softmax(pos_bias + decoder_bias) - epoch {epoch}", fontsize=11)
    fig2.tight_layout()

    out_base = args.out or "posaware_bias"
    out1 = f"{out_base}_gaussian.png"
    out2 = f"{out_base}_profile.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out1}")
    print(f"Saved {out2}")
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
