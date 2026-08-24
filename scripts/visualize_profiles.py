"""Visualize Hermite basis, learned decoder, and profile predictions.

Usage:
    # View Hermite basis and bias from a basis file
    uv run python scripts/visualize_profiles.py --basis /path/to/profile_basis_20.pt

    # View learned decoder W and bias from a checkpoint
    uv run python scripts/visualize_profiles.py --checkpoint /path/to/last.ckpt

    # View predicted profiles vs observed shoeboxes
    uv run python scripts/visualize_profiles.py --checkpoint /path/to/last.ckpt \
        --data-dir /path/to/pytorch_data --n-samples 16

    # All three together
    uv run python scripts/visualize_profiles.py \
        --basis /path/to/profile_basis_20.pt \
        --checkpoint /path/to/last.ckpt \
        --data-dir /path/to/pytorch_data
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize profile basis and predictions"
    )
    parser.add_argument(
        "--basis", type=Path, default=None, help="Path to profile_basis_*.pt"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Path to .ckpt file"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data dir with counts/masks",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of random reflections",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output prefix (saves multiple PNGs)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for sampling"
    )
    return parser.parse_args()


def _infer_shape(n_pixels):
    """Guess (H, W) or (D, H, W) from pixel count."""
    sqrt = int(n_pixels**0.5)
    if sqrt * sqrt == n_pixels:
        return (sqrt, sqrt)
    for d in [3, 5, 7]:
        hw = n_pixels // d
        s = int(hw**0.5)
        if d * s * s == n_pixels:
            return (d, s, s)
    return (n_pixels,)


def plot_basis(basis_path, out_prefix):
    """Plot the Hermite/PCA basis vectors W and bias b."""
    basis = torch.load(basis_path, map_location="cpu", weights_only=False)
    W = basis["W"]  # (K, d)
    b = basis["b"]  # (K,)
    d = W.shape[1]
    shape = _infer_shape(W.shape[0])
    orders = basis.get("orders", None)

    # Plot bias (reference profile)
    fig_bias, ax = plt.subplots(1, 1, figsize=(4, 4))
    ref_profile = F.softmax(b, dim=0).reshape(
        shape[-2:] if len(shape) >= 2 else shape
    )
    im = ax.imshow(ref_profile.numpy(), cmap="viridis", origin="lower")
    ax.set_title("Bias b (reference profile)")
    fig_bias.colorbar(im, ax=ax, shrink=0.8)
    fig_bias.tight_layout()
    fname = f"{out_prefix}_bias.png" if out_prefix else "basis_bias.png"
    fig_bias.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig_bias)

    # Plot basis vectors
    ncols = min(d, 7)
    nrows = (d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    if d == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for i in range(d):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        mode = W[:, i].reshape(shape[-2:] if len(shape) >= 2 else shape)
        vmax = mode.abs().max().item()
        ax.imshow(
            mode.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower"
        )
        label = (
            f"({orders[i][0]},{orders[i][1]})"
            if orders and len(orders[i]) == 2
            else str(i)
        )
        ax.set_title(f"W[:,{i}] {label}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(d, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Hermite basis: {d} modes, shape {shape}", fontsize=11)
    fig.tight_layout()
    fname = f"{out_prefix}_basis.png" if out_prefix else "basis_modes.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_learned_decoder(checkpoint_path, out_prefix):
    """Plot the learned decoder W and bias from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Find decoder keys (learned_basis_profile stores in surrogates.qp.decoder)
    W_key = "surrogates.qp.decoder.weight"
    b_key = "surrogates.qp.decoder.bias"

    if W_key not in sd:
        print(f"No learned decoder found (missing {W_key})")
        return

    # decoder is nn.Linear(d, K), so weight is (K, d)
    W = sd[W_key]  # (K, d)
    b = sd[b_key]  # (K,)
    d = W.shape[1]
    shape = _infer_shape(W.shape[0])
    epoch = ckpt.get("epoch", "?")

    # Plot bias
    fig_bias, ax = plt.subplots(1, 1, figsize=(4, 4))
    ref_profile = F.softmax(b, dim=0).reshape(
        shape[-2:] if len(shape) >= 2 else shape
    )
    im = ax.imshow(ref_profile.numpy(), cmap="viridis", origin="lower")
    ax.set_title(f"Learned bias (epoch {epoch})")
    fig_bias.colorbar(im, ax=ax, shrink=0.8)
    fig_bias.tight_layout()
    fname = (
        f"{out_prefix}_learned_bias.png" if out_prefix else "learned_bias.png"
    )
    fig_bias.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig_bias)

    # Plot learned W columns
    ncols = min(d, 7)
    nrows = (d + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    if d == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for i in range(d):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        mode = W[:, i].reshape(shape[-2:] if len(shape) >= 2 else shape)
        vmax = mode.abs().max().item()
        ax.imshow(
            mode.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower"
        )
        ax.set_title(f"W[:,{i}]", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(d, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"Learned decoder: {d} modes, shape {shape} (epoch {epoch})",
        fontsize=11,
    )
    fig.tight_layout()
    fname = (
        f"{out_prefix}_learned_modes.png"
        if out_prefix
        else "learned_modes.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def plot_profile_comparison(
    checkpoint_path, data_dir, n_samples, seed, out_prefix
):
    """Compare predicted profiles vs observed shoeboxes."""
    import numpy as np

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    epoch = ckpt.get("epoch", "?")

    W_key = "surrogates.qp.decoder.weight"
    b_key = "surrogates.qp.decoder.bias"
    if W_key not in sd:
        # Try fixed basis
        W_key = "surrogates.qp.W"
        b_key = "surrogates.qp.b"
    if W_key not in sd:
        print("No profile decoder found in checkpoint")
        return

    W = sd[W_key]  # (K, d)
    b = sd[b_key]  # (K,)
    K = W.shape[0]
    shape = _infer_shape(K)
    spatial = shape[-2:] if len(shape) >= 2 else shape

    # Load data
    data_dir = Path(data_dir)
    counts_path = data_dir / "counts.npy"
    masks_path = data_dir / "masks.npy"
    if counts_path.exists():
        counts = (
            torch.from_numpy(np.load(counts_path, mmap_mode="r")[:])
            .float()
            .squeeze(-1)
        )
        masks = (
            torch.from_numpy(np.load(masks_path, mmap_mode="r")[:])
            .float()
            .squeeze(-1)
        )
    else:
        counts = (
            torch.load(data_dir / "counts.pt", weights_only=False)
            .float()
            .squeeze(-1)
        )
        masks = (
            torch.load(data_dir / "masks.pt", weights_only=False)
            .float()
            .squeeze(-1)
        )

    # Load predictions if available - look for parquet or construct from checkpoint
    # For simplicity, we reconstruct profiles from the model directly
    from integrator.utils.factory_utils import (
        construct_data_loader,
        construct_integrator,
    )

    # Find config
    meta_path = None
    for parent in checkpoint_path.parents:
        candidates = list(parent.glob("run_paths.yaml"))
        if candidates:
            meta_path = candidates[0]
            break

    if meta_path is None:
        # Try to find config_log.yaml near the checkpoint
        print(
            "Could not find run_paths.yaml - using raw shoebox comparison only"
        )
        _plot_raw_shoeboxes(
            counts, masks, spatial, n_samples, seed, epoch, out_prefix
        )
        return

    import yaml

    meta = yaml.safe_load(meta_path.read_text())
    config_path = meta.get("config")
    if config_path is None or not Path(config_path).exists():
        print(f"Config not found at {config_path}")
        _plot_raw_shoeboxes(
            counts, masks, spatial, n_samples, seed, epoch, out_prefix
        )
        return

    from integrator.utils import inject_binning_labels, load_config

    cfg = load_config(config_path)
    data_loader = construct_data_loader(cfg)
    data_loader.setup()
    inject_binning_labels(data_loader, cfg)

    integrator = construct_integrator(cfg, skip_warmstart=True)
    integrator.load_state_dict(sd)
    integrator.eval()

    # Get a batch
    torch.manual_seed(seed)
    all_idx = torch.randperm(len(data_loader.full_dataset))[:n_samples]
    batch_data = [data_loader.full_dataset[i] for i in all_idx]
    from torch.utils.data import DataLoader

    loader = DataLoader(
        [data_loader.full_dataset[i] for i in all_idx],
        batch_size=n_samples,
        shuffle=False,
        collate_fn=data_loader.full_dataset.collate_fn
        if hasattr(data_loader.full_dataset, "collate_fn")
        else None,
    )

    batch = next(iter(loader))
    counts_batch, shoebox_batch, masks_batch, metadata = batch

    with torch.no_grad():
        outputs = integrator(
            counts_batch, shoebox_batch, masks_batch, metadata
        )

    qp = outputs["qp"]
    pred_profiles = qp.mean_profile  # (B, K)

    # Normalize observed
    obs = (counts_batch * masks_batch).float()
    obs_sum = obs.sum(dim=-1, keepdim=True).clamp(min=1)
    obs_norm = obs / obs_sum

    # Plot grid
    ncols = min(n_samples, 4)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols * 2, figsize=(3 * ncols * 2, 3 * nrows)
    )
    if nrows == 1:
        axes = [axes]

    for i in range(n_samples):
        r = i // ncols
        c = (i % ncols) * 2

        obs_img = obs_norm[i].reshape(spatial).numpy()
        pred_img = pred_profiles[i].reshape(spatial).detach().numpy()

        vmax = max(obs_img.max(), pred_img.max())

        ax_obs = axes[r][c]
        ax_pred = axes[r][c + 1]

        ax_obs.imshow(
            obs_img, cmap="viridis", vmin=0, vmax=vmax, origin="lower"
        )
        ax_obs.set_title("observed", fontsize=8)
        ax_obs.set_xticks([])
        ax_obs.set_yticks([])

        ax_pred.imshow(
            pred_img, cmap="viridis", vmin=0, vmax=vmax, origin="lower"
        )
        ax_pred.set_title("predicted", fontsize=8)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

    # Hide unused
    for i in range(n_samples, nrows * ncols):
        r = i // ncols
        c = (i % ncols) * 2
        axes[r][c].set_visible(False)
        axes[r][c + 1].set_visible(False)

    fig.suptitle(
        f"Profile comparison: observed vs predicted (epoch {epoch})",
        fontsize=11,
    )
    fig.tight_layout()
    fname = (
        f"{out_prefix}_profiles.png"
        if out_prefix
        else "profile_comparison.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def _plot_raw_shoeboxes(
    counts, masks, spatial, n_samples, seed, epoch, out_prefix
):
    """Fallback: just plot random shoeboxes without model predictions."""
    torch.manual_seed(seed)
    idx = torch.randperm(len(counts))[:n_samples]

    obs = (counts[idx] * masks[idx]).float()
    obs_sum = obs.sum(dim=-1, keepdim=True).clamp(min=1)
    obs_norm = obs / obs_sum

    ncols = min(n_samples, 4)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for i in range(n_samples):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        img = obs_norm[i].reshape(spatial).numpy()
        ax.imshow(img, cmap="viridis", origin="lower")
        ax.set_title(f"refl {idx[i].item()}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_samples, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Raw shoeboxes (normalized, epoch {epoch})")
    fig.tight_layout()
    fname = (
        f"{out_prefix}_shoeboxes.png" if out_prefix else "raw_shoeboxes.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


def main():
    args = parse_args()
    out = args.out or "profile"

    if args.basis:
        plot_basis(args.basis, out)

    if args.checkpoint:
        plot_learned_decoder(args.checkpoint, out)

    if args.checkpoint and args.data_dir:
        plot_profile_comparison(
            args.checkpoint, args.data_dir, args.n_samples, args.seed, out
        )

    if not args.basis and not args.checkpoint:
        print("Provide --basis and/or --checkpoint. See --help.")


if __name__ == "__main__":
    main()
