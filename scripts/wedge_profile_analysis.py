"""Empirical per-wedge profile analysis — decides whether per-bin basis is worth building.

Bins reflections by azimuthal angle around the beam center (on the detector),
averages bg-subtracted profiles within each wedge, and produces:

1. Per-wedge average profiles as a grid of heatmaps (middle z-slice)
2. Pairwise cosine similarity matrix across wedges (ordered by azimuth)
3. 2-fold symmetry test: cos(P_i, rotate_180(P_{i+n/2})) for each wedge pair
4. Variance decomposition: how much total profile variance lives between wedges
   vs within-wedge individual-reflection variation

Interpretation:
- If pairwise cos sim between ALL wedges is > 0.95, wedges are basically
  identical — no per-bin basis needed.
- If near-wedge cos sim is high but far-wedge cos sim is low, there's
  genuine azimuthal structure — per-bin basis could help.
- If 2-fold symmetry test scores > 0.95 uniformly, you can fold to N/2 wedges.

Run:
    uv run python scripts/wedge_profile_analysis.py \\
        --data-dir /Users/luis/from_harvard_rc/hewl_9b7c \\
        --n-wedges 16 \\
        --n-subsample 200000
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from integrator.utils.prepare_priors import _bg_subtract_signal


def _wedge_labels(metadata: dict, n_wedges: int) -> torch.Tensor:
    """Bin reflections by azimuthal angle on the detector.

    Uses predicted pixel positions `xyzcal.px.{0,1}` centered at their mean
    as the proxy for beam center. Azimuth = atan2(dy, dx) in (-π, π],
    then discretized into n_wedges equal-angle bins in [0, n_wedges).
    """
    x = metadata["xyzcal.px.0"].float()
    y = metadata["xyzcal.px.1"].float()
    dx = x - x.mean()
    dy = y - y.mean()
    theta = torch.atan2(dy, dx)  # (-π, π]
    normalized = (theta + torch.pi) / (2 * torch.pi)  # [0, 1)
    idx = (normalized * n_wedges).long().clamp(0, n_wedges - 1)
    return idx


def _normalize_profile(signal: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Laplace-smoothed per-reflection proportions."""
    totals = signal.sum(dim=1, keepdim=True)
    K = signal.shape[1]
    return (signal + eps) / (totals + eps * K)


def _rotate_180_xyz(vol: np.ndarray) -> np.ndarray:
    """Full 180° rotation of a (D, H, W) profile — flips all three axes."""
    return vol[::-1, ::-1, ::-1].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--counts-file", default="counts.pt")
    parser.add_argument("--masks-file", default="masks.pt")
    parser.add_argument("--metadata-file", default="metadata.pt")
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--H", type=int, default=21)
    parser.add_argument("--W", type=int, default=21)
    parser.add_argument("--n-wedges", type=int, default=16)
    parser.add_argument("--n-subsample", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--laplace-alpha", type=float, default=1.0)
    parser.add_argument(
        "--output", type=str, default="wedge_profile_analysis.png"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading metadata: {data_dir / args.metadata_file}")
    metadata = torch.load(
        data_dir / args.metadata_file,
        weights_only=False,
        map_location="cpu",
    )
    N_total = metadata["xyzcal.px.0"].shape[0]
    print(f"  N reflections: {N_total:,}")

    print(f"Loading counts (mmap): {data_dir / args.counts_file}")
    counts = torch.load(
        data_dir / args.counts_file,
        weights_only=False,
        mmap=True,
    )
    print(f"Loading masks (mmap): {data_dir / args.masks_file}")
    masks = torch.load(
        data_dir / args.masks_file,
        weights_only=False,
        mmap=True,
    )

    # Assign wedge labels using full metadata (cheap, just atan2)
    wedge_all = _wedge_labels(metadata, args.n_wedges)
    counts_per_wedge_all = torch.bincount(wedge_all, minlength=args.n_wedges)
    print("\nWedge occupancy (full dataset):")
    for i, c in enumerate(counts_per_wedge_all.tolist()):
        print(f"  wedge {i:2d}: {c:>9,}")

    # Stratified subsample: roughly equal reflections per wedge
    per_wedge_cap = args.n_subsample // args.n_wedges
    g = torch.Generator().manual_seed(args.seed)
    selected_idx_list = []
    for k in range(args.n_wedges):
        idx_k = (wedge_all == k).nonzero(as_tuple=True)[0]
        n_k = min(per_wedge_cap, idx_k.numel())
        if n_k == 0:
            continue
        chosen = idx_k[torch.randperm(idx_k.numel(), generator=g)[:n_k]]
        selected_idx_list.append(chosen)
    selected_idx = torch.cat(selected_idx_list)
    selected_idx, _ = torch.sort(
        selected_idx
    )  # sequential = friendlier to mmap
    print(
        f"\nStratified subsample: {selected_idx.numel():,} reflections "
        f"(~{per_wedge_cap} per wedge)"
    )

    print("Materializing subset counts/masks...")
    counts_sub = counts[selected_idx].clone()
    masks_sub = masks[selected_idx].clone()
    wedge_sub = wedge_all[selected_idx]

    print("Background-subtracting...")
    signal = _bg_subtract_signal(counts_sub, masks_sub, args.D, args.H, args.W)

    # Drop reflections with no signal
    totals = signal.sum(dim=1)
    keep = totals > 0
    signal = signal[keep]
    wedge_sub = wedge_sub[keep]
    print(f"  {int((~keep).sum())} reflections dropped (zero total signal)")

    # Per-reflection normalized profiles (proportion space with Laplace smoothing)
    print(f"Computing proportions (laplace_alpha={args.laplace_alpha})...")
    proportions = _normalize_profile(signal, eps=args.laplace_alpha)

    # Per-wedge mean profile (in proportion space, on log scale)
    K = args.D * args.H * args.W
    mean_profile = torch.zeros(args.n_wedges, K)
    n_per_wedge = torch.zeros(args.n_wedges)
    for k in range(args.n_wedges):
        mask_k = wedge_sub == k
        n_per_wedge[k] = mask_k.sum()
        if mask_k.sum() > 0:
            # Mean of LOG proportions — this is the natural "basis b"
            # each wedge would have if used for softmax(b_k) = p_bar_k
            log_p = torch.log(proportions[mask_k])
            mean_profile[k] = log_p.mean(dim=0)

    # Cosine similarity between mean profiles (centered by global mean log-proportion)
    global_b = mean_profile.mean(dim=0)
    centered = mean_profile - global_b  # (n_wedges, K)
    norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-12)
    unit = centered / norms
    sim = (unit @ unit.T).numpy()  # (n_wedges, n_wedges)

    # Variance decomposition
    total_var = (proportions.log() - global_b).pow(2).sum().item()
    between_var = (centered.pow(2).sum(dim=1) * n_per_wedge).sum().item()
    within_var = total_var - between_var
    between_frac = between_var / max(total_var, 1e-12)
    print("\nVariance decomposition (log-proportion space):")
    print(f"  total sum-of-squares:  {total_var:.2e}")
    print(
        f"  between-wedge SS:      {between_var:.2e}  ({between_frac * 100:.2f}% of total)"
    )
    print(
        f"  within-wedge SS:       {within_var:.2e}  ({(1 - between_frac) * 100:.2f}% of total)"
    )
    if between_frac < 0.02:
        print(
            "  → between-wedge variation is < 2% of total; per-bin basis likely wasted."
        )
    elif between_frac < 0.10:
        print(
            "  → modest between-wedge variation (2-10%). Per-bin basis may help at the margin."
        )
    else:
        print(
            "  → substantial between-wedge variation (> 10%). Per-bin basis promising."
        )

    # 2-fold symmetry test: wedge i vs wedge (i + n/2) after rotating 180° in xyz
    print(
        "\n2-fold symmetry test "
        "(cos sim between wedge_i and rotate_180(wedge_(i+n/2))):"
    )
    half = args.n_wedges // 2
    sym_cos = []
    for i in range(half):
        j = i + half
        pi = centered[i].view(args.D, args.H, args.W).numpy()
        pj = centered[j].view(args.D, args.H, args.W).numpy()
        pj_rot = _rotate_180_xyz(pj)
        dot = float((pi * pj_rot).sum())
        ni = float(np.linalg.norm(pi))
        nj = float(np.linalg.norm(pj_rot))
        c = dot / max(ni * nj, 1e-12)
        sym_cos.append(c)
        print(f"  wedge {i:2d} vs rotate_180(wedge {j:2d}): cos = {c:+.4f}")
    sym_cos_arr = np.array(sym_cos)
    print(f"  mean: {sym_cos_arr.mean():+.3f}   min: {sym_cos_arr.min():+.3f}")
    if sym_cos_arr.mean() > 0.9:
        print(
            "  → strong 2-fold symmetry — you can likely halve to n_wedges/2."
        )
    elif sym_cos_arr.mean() > 0.5:
        print("  → partial 2-fold symmetry — folding is debatable.")
    else:
        print("  → no 2-fold symmetry — keep full n_wedges.")

    # Cross-wedge |cos| summary
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    abs_off = np.abs(off[~np.isnan(off)])
    print(
        "\nPairwise cosine similarity across wedges "
        "(centered mean log-profiles):"
    )
    print(f"  mean |cos|: {abs_off.mean():.3f}")
    print(f"  min cos:    {off[~np.isnan(off)].min():+.3f}")
    print(f"  max cos:    {off[~np.isnan(off)].max():+.3f}")

    # -- Plot --
    fig = plt.figure(figsize=(22, 14))
    D, H, W = args.D, args.H, args.W
    mid = D // 2

    # Row 1-2: per-wedge mean profiles (middle slice of softmax)
    ncols = args.n_wedges // 2 + args.n_wedges % 2
    gs = fig.add_gridspec(
        4,
        ncols,
        height_ratios=[1.0, 1.0, 1.4, 1.4],
        hspace=0.5,
        wspace=0.15,
    )

    # Profiles as softmax over K-dim mean log-profile, then take middle slice
    prof_imgs = []
    for k in range(args.n_wedges):
        # Take middle slice of the log-proportion mean, convert to profile
        p_k = torch.softmax(mean_profile[k], dim=-1).view(D, H, W).numpy()
        prof_imgs.append(p_k[mid])
    vmax = max(img.max() for img in prof_imgs)

    for k in range(args.n_wedges):
        row = 0 if k < ncols else 1
        col = k if k < ncols else k - ncols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(prof_imgs[k], cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(
            f"wedge {k}\n(N={int(n_per_wedge[k])})",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 3: pairwise similarity matrix
    ax_sim = fig.add_subplot(gs[2, : ncols // 2 + 1])
    im = ax_sim.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax_sim.set_title(
        "Pairwise cos similarity (centered mean log-profile)", fontsize=10
    )
    ax_sim.set_xlabel("wedge j")
    ax_sim.set_ylabel("wedge i")
    plt.colorbar(im, ax=ax_sim, fraction=0.04)

    # Row 3 right: 2-fold symmetry plot
    ax_sym = fig.add_subplot(gs[2, ncols // 2 + 1 :])
    ax_sym.bar(
        range(half),
        sym_cos_arr,
        color=["C0" if c > 0.5 else "C3" for c in sym_cos_arr],
    )
    ax_sym.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax_sym.axhline(0.9, color="green", linestyle="--", alpha=0.5)
    ax_sym.set_xlabel(f"wedge i (pairing with wedge i+{half} rotated 180°)")
    ax_sym.set_ylabel("cos similarity")
    ax_sym.set_title(
        f"2-fold rotational symmetry test (mean={sym_cos_arr.mean():+.3f})",
        fontsize=10,
    )
    ax_sym.set_ylim(-1, 1)
    ax_sym.set_xticks(range(half))

    # Row 4: off-diagonal cos distribution (histogram)
    ax_hist = fig.add_subplot(gs[3, :])
    off_all = off[~np.isnan(off)]
    ax_hist.hist(off_all, bins=40, color="C2", alpha=0.7)
    ax_hist.axvline(0, color="gray", linestyle="--")
    ax_hist.set_xlabel("cos similarity between wedge pairs")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(
        f"Distribution of off-diagonal pairwise cos sim  —  "
        f"mean |cos|={abs_off.mean():.3f},  between-wedge variance="
        f"{between_frac * 100:.2f}% of total",
        fontsize=10,
    )

    fig.suptitle(
        f"Per-wedge profile analysis  ({args.n_wedges} wedges, "
        f"{int(n_per_wedge.sum()):,} reflections)",
        fontsize=12,
        y=0.99,
    )
    out = Path(args.output).resolve()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
