"""Visualize learned profile basis W and bias b from a checkpoint.

Usage:
    # Monochromatic (3D: D*H*W, shows middle slice)
    uv run python scripts/visualize_learned_basis.py \
        --ckpt path/to/checkpoint.ckpt \
        --dims 3 21 21

    # Polychromatic (2D: H*W)
    uv run python scripts/visualize_learned_basis.py \
        --ckpt path/to/checkpoint.ckpt \
        --dims 25 25

    # With warmstart comparison
    uv run python scripts/visualize_learned_basis.py \
        --ckpt path/to/checkpoint.ckpt \
        --dims 3 21 21 \
        --warmstart path/to/hermite_profile_basis.pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


def load_basis_from_ckpt(ckpt_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["state_dict"]

    W_key = [k for k in sd if "surrogates.qp.decoder.weight" in k]
    b_key = [k for k in sd if "surrogates.qp.decoder.bias" in k]

    if not W_key or not b_key:
        raise KeyError(
            f"Could not find decoder weight/bias in checkpoint. "
            f"Available keys with 'qp': {[k for k in sd if 'qp' in k]}"
        )

    W = sd[W_key[0]].float()  # (K, d)
    b = sd[b_key[0]].float()  # (K,)
    return W, b


def load_basis_from_pt(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.load(path, weights_only=False, map_location="cpu")
    return basis["W"].float(), basis["b"].float()


def basis_to_profiles(
    W: torch.Tensor, b: torch.Tensor, scale: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert decoder (W, b) to profile perturbation images.

    Each basis column is shown as softmax(scale * W[:,i] + b) - softmax(b),
    i.e. the change in profile when latent h_i = scale (vs h=0).

    Returns:
        ref_profile: softmax(b) - the reference profile (h=0)
        basis_perturbations: softmax(scale * W[:,i] + b) - ref for each i
    """
    ref_profile = F.softmax(b, dim=0)

    d = W.shape[1]
    perturbations = []
    for i in range(d):
        logits = scale * W[:, i] + b
        perturbations.append(F.softmax(logits, dim=0) - ref_profile)
    basis_perturbations = torch.stack(perturbations)  # (d, K)
    return ref_profile, basis_perturbations


def reshape_profile(profile: torch.Tensor, dims: list[int]) -> np.ndarray:
    """Reshape flat profile to spatial dims, take middle slice if 3D."""
    vol = profile.reshape(*dims).numpy()
    if len(dims) == 3:
        mid = dims[0] // 2
        return vol[mid]
    return vol


def plot_basis(
    W: torch.Tensor,
    b: torch.Tensor,
    dims: list[int],
    title_prefix: str = "",
    W_init: torch.Tensor | None = None,
    b_init: torch.Tensor | None = None,
):
    ref_profile, basis_perturbations = basis_to_profiles(W, b)
    d = W.shape[1]

    has_init = W_init is not None and b_init is not None
    if has_init:
        ref_init, basis_init_perturbations = basis_to_profiles(W_init, b_init)
        d_init = W_init.shape[1]

    # --- Reference profile (bias only) ---
    if has_init:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(reshape_profile(ref_init, dims), cmap="magma")
        axes[0].set_title("Reference profile (init)")
        axes[0].axis("off")
        axes[1].imshow(reshape_profile(ref_profile, dims), cmap="magma")
        axes[1].set_title("Reference profile (learned)")
        axes[1].axis("off")
        fig.suptitle(f"{title_prefix}Reference profile: softmax(b)")
        plt.tight_layout()
        plt.savefig("reference_profile.png", dpi=150, bbox_inches="tight")
        plt.show()

        # Bias difference
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        diff = (ref_profile - ref_init).numpy()
        im = ax.imshow(
            diff.reshape(*dims) if len(dims) == 2
            else diff.reshape(*dims)[dims[0] // 2],
            cmap="RdBu_r", vmin=-abs(diff).max(), vmax=abs(diff).max(),
        )
        ax.set_title("Δ reference profile (learned − init)")
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig("reference_profile_diff.png", dpi=150, bbox_inches="tight")
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.imshow(reshape_profile(ref_profile, dims), cmap="magma")
        ax.set_title(f"{title_prefix}Reference profile: softmax(b)")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig("reference_profile.png", dpi=150, bbox_inches="tight")
        plt.show()

    # --- Basis perturbations: how each latent dim deforms the profile ---
    ncols = min(8, d)
    nrows = (d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.array(axes).flatten()
    global_vmax = basis_perturbations.abs().max().item()
    for i in range(d):
        img = reshape_profile(basis_perturbations[i], dims)
        axes[i].imshow(img, cmap="RdBu_r", vmin=-global_vmax, vmax=global_vmax)
        axes[i].set_title(f"h[{i}] = 3", fontsize=8)
        axes[i].axis("off")
    for i in range(d, len(axes)):
        axes[i].axis("off")
    fig.suptitle(
        f"{title_prefix}Basis perturbations: softmax(3·W[:,i] + b) − softmax(b)"
    )
    plt.tight_layout()
    plt.savefig("basis_perturbations.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- If warmstart: show how each column changed from init ---
    if has_init:
        d_compare = min(d, d_init)
        ncols_c = min(8, d_compare)
        nrows_c = (d_compare + ncols_c - 1) // ncols_c
        fig, axes = plt.subplots(
            nrows_c, ncols_c, figsize=(2.5 * ncols_c, 2.5 * nrows_c)
        )
        axes = np.array(axes).flatten()
        diff_vmax = 0.0
        diffs = []
        for i in range(d_compare):
            diff = (basis_perturbations[i] - basis_init_perturbations[i]).numpy()
            diffs.append(diff)
            diff_vmax = max(diff_vmax, abs(diff).max())
        for i in range(d_compare):
            img = diffs[i].reshape(*dims) if len(dims) == 2 \
                else diffs[i].reshape(*dims)[dims[0] // 2]
            axes[i].imshow(img, cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax)
            axes[i].set_title(f"Δ col {i}", fontsize=8)
            axes[i].axis("off")
        for i in range(d_compare, len(axes)):
            axes[i].axis("off")
        fig.suptitle(f"{title_prefix}Perturbation change: learned − init")
        plt.tight_layout()
        plt.savefig("basis_perturbations_diff.png", dpi=150, bbox_inches="tight")
        plt.show()

    # --- Raw W columns as heatmaps (logit space, not probability) ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.array(axes).flatten()
    for i in range(d):
        col = W[:, i].numpy()
        img = col.reshape(*dims) if len(dims) == 2 \
            else col.reshape(*dims)[dims[0] // 2]
        vmax = abs(col).max()
        axes[i].imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"W[:,{i}] (logit)", fontsize=8)
        axes[i].axis("off")
    for i in range(d, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"{title_prefix}Raw basis columns W (logit space)")
    plt.tight_layout()
    plt.savefig("basis_logits.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Summary stats ---
    print(f"Decoder shape: W {tuple(W.shape)}, b {tuple(b.shape)}")
    print(f"W range: [{W.min():.4f}, {W.max():.4f}], std: {W.std():.4f}")
    print(f"b range: [{b.min():.4f}, {b.max():.4f}]")
    print(f"Reference profile peak: {ref_profile.max():.6f}")
    if has_init:
        W_delta = W[:, :d_compare] - W_init[:, :d_compare]
        b_delta = b - b_init
        print(f"ΔW Frobenius norm: {W_delta.norm():.4f}")
        print(f"Δb L2 norm: {b_delta.norm():.4f}")
        print(
            f"ΔW per-column norms: "
            + ", ".join(f"{W_delta[:, i].norm():.3f}" for i in range(min(d_compare, 10)))
        )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned profile basis from checkpoint"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--dims", type=int, nargs="+", required=True,
        help="Profile spatial dims, e.g. '3 21 21' (mono) or '25 25' (poly)",
    )
    parser.add_argument(
        "--warmstart", type=str, default=None,
        help="Path to warmstart basis .pt file for comparison",
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Title prefix for plots",
    )
    args = parser.parse_args()

    W, b = load_basis_from_ckpt(args.ckpt)
    print(f"Loaded checkpoint: {args.ckpt}")

    W_init, b_init = None, None
    if args.warmstart:
        W_init, b_init = load_basis_from_pt(args.warmstart)
        print(f"Loaded warmstart basis: {args.warmstart}")

    plot_basis(W, b, args.dims, args.prefix, W_init, b_init)


if __name__ == "__main__":
    main()
