"""Build a PCA profile basis from a dataset of shoeboxes.

Streams (via mmap) from counts.pt + masks.pt, subsamples, bg-subtracts, and
runs SVD on log(proportions) to produce a PCA basis compatible with
FixedBasisProfileSurrogate.

Output: pca_profile_basis.pt with keys W, b, d, explained_var, sigma_prior,
basis_type.

Run:
    uv run python scripts/build_pca_basis.py \\
        --data-dir /Users/luis/from_harvard_rc/hewl_9b7c \\
        --d 14
"""

import argparse
from pathlib import Path

import torch

from integrator.utils.prepare_priors import _bg_subtract_signal


def build_pca_basis(
    signal: torch.Tensor,
    d: int,
    laplace_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PCA of log-proportions with Laplace smoothing.

    Laplace smoothing: p_i = (s_i + alpha) / (sum(s) + alpha*K).
    With alpha > 0, log(p) is bounded regardless of which pixels are
    zero in the raw signal. This is critical when the mask zeroes out
    a large fraction of pixels — raw log(clamp(0, 1e-8)) injects huge
    artificial variance that dominates the first few PCs.
    """
    K = signal.shape[1]
    totals = signal.sum(dim=1, keepdim=True)
    proportions = (signal + laplace_alpha) / (totals + laplace_alpha * K)
    log_profiles = torch.log(proportions)

    b = log_profiles.mean(dim=0)
    centered = log_profiles - b

    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    d_actual = min(d, S.shape[0])
    W = Vh[:d_actual].T  # (K, d_actual)

    total_var = (S**2).sum()
    explained_var = (S[:d_actual] ** 2) / total_var
    return W.float(), b.float(), explained_var.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--counts-file", type=str, default="counts.pt")
    parser.add_argument("--masks-file", type=str, default="masks.pt")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--H", type=int, default=21)
    parser.add_argument("--W", type=int, default=21)
    parser.add_argument(
        "--d", type=int, default=14, help="number of principal components"
    )
    parser.add_argument(
        "--n-subsample",
        type=int,
        default=100_000,
        help="number of reflections to sample for PCA "
        "(full dataset if larger or 0)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--laplace-alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing strength for log-proportions",
    )
    parser.add_argument(
        "--output-file", type=str, default="pca_profile_basis.pt"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    counts_path = data_dir / args.counts_file
    masks_path = data_dir / args.masks_file
    out_path = data_dir / args.output_file

    print(f"Loading {counts_path} (mmap)...")
    counts = torch.load(counts_path, weights_only=False, mmap=True)
    print(f"Loading {masks_path} (mmap)...")
    masks = torch.load(masks_path, weights_only=False, mmap=True)

    N_total = counts.shape[0]
    K_expected = args.D * args.H * args.W
    assert counts.shape[1] == K_expected, (
        f"counts shape {counts.shape[1]} != D*H*W = {K_expected}"
    )
    print(
        f"Dataset: N={N_total:,} reflections, K={K_expected} pixels "
        f"(D={args.D}, H={args.H}, W={args.W})"
    )

    # Subsample
    n = min(args.n_subsample, N_total) if args.n_subsample > 0 else N_total
    g = torch.Generator().manual_seed(args.seed)
    if n < N_total:
        idx = torch.randperm(N_total, generator=g)[:n]
        idx, _ = torch.sort(idx)  # sequential access is friendlier to mmap
        print(f"Subsampling {n:,}/{N_total:,} reflections...")
    else:
        idx = torch.arange(N_total)
        print(f"Using all {n:,} reflections...")

    counts_sub = counts[idx].clone()
    masks_sub = masks[idx].clone()
    print(
        f"Subset materialized: counts={counts_sub.shape}, "
        f"masks={masks_sub.shape}"
    )

    print("Background-subtracting...")
    signal = _bg_subtract_signal(counts_sub, masks_sub, args.D, args.H, args.W)
    print(
        f"  signal shape: {signal.shape}, "
        f"nonzero fraction: {(signal > 0).float().mean().item():.3f}"
    )

    # Drop reflections with no signal (sum == 0) to avoid log(0) dominating
    totals = signal.sum(dim=1)
    keep = totals > 0
    dropped = int((~keep).sum())
    if dropped > 0:
        print(f"  dropping {dropped} reflections with zero total signal")
        signal = signal[keep]

    print(
        f"Running SVD on ({signal.shape[0]:,} × {signal.shape[1]}) "
        f"log-profile matrix (laplace_alpha={args.laplace_alpha})..."
    )
    W, b, ev = build_pca_basis(
        signal, args.d, laplace_alpha=args.laplace_alpha
    )
    print(f"  d = {W.shape[1]}, explained variance:")
    for j, v in enumerate(ev.tolist()):
        print(f"    PC{j}: {v:.1%}")
    print(f"  cumulative EV: {ev.sum().item():.1%}")

    # sigma_prior = global std of latent codes (for fallback).
    # Must use the same Laplace smoothing as the basis fit, otherwise the
    # reconstruction residuals are artificially huge.
    K = signal.shape[1]
    totals_f = signal.sum(dim=1, keepdim=True)
    proportions = (signal + args.laplace_alpha) / (
        totals_f + args.laplace_alpha * K
    )
    log_profiles = torch.log(proportions)
    centered = log_profiles - b
    h_all = centered @ W  # (N, d)
    sigma_prior = float(h_all.std().item())
    sigma_prior = max(sigma_prior, 1.0)
    print(f"  sigma_prior (floor 1.0): {sigma_prior:.3f}")

    result = {
        "W": W,
        "b": b,
        "d": int(W.shape[1]),
        "explained_var": ev,
        "sigma_prior": sigma_prior,
        "basis_type": "pca",
        "n_reflections_used": int(signal.shape[0]),
        "shoebox_shape": (args.D, args.H, args.W),
        "laplace_alpha": args.laplace_alpha,
    }
    torch.save(result, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
