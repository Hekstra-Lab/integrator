"""Measure per-HKL intensity consistency across observations.

For each unique HKL, computes the spread of qi_mean values across
redundant observations. A better model should produce more consistent
intensities for the same reflection.

Metrics:
  - CV (coefficient of variation) = std/mean per HKL
  - MAD/median (robust alternative)
  - Weighted R_merge = sigma|I_i - <I>| / sigma I_i

Usage:
    # Compare two runs
    uv run python scripts/hkl_consistency.py \
        --run-dir /path/to/run_A --run-dir /path/to/run_B \
        --metadata /path/to/metadata.pt \
        --labels "old model" "new model"

    # Single run, specific epoch
    uv run python scripts/hkl_consistency.py \
        --run-dir /path/to/run \
        --metadata /path/to/metadata.pt \
        --epoch 99

Options:
    --epoch     Which epoch (default: latest)
    --out       Output directory
    --min-mult  Minimum multiplicity to include an HKL (default: 3)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-HKL intensity consistency analysis"
    )
    p.add_argument(
        "--run-dir", type=Path, nargs="+", required=True,
        help="One or more run directories to compare",
    )
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-mult", type=int, default=3)
    p.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels for each run (default: directory names)",
    )
    return p.parse_args()


def resolve_wandb_dir(run_dir):
    meta_path = run_dir / "run_paths.yaml"
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    return Path(meta["wandb"]["log_dir"])


def load_predictions(run_dir, epoch=None):
    wandb_dir = resolve_wandb_dir(run_dir)
    pred_dir = (
        wandb_dir.parent / "predictions"
        if wandb_dir.name == "files"
        else wandb_dir / "predictions"
    )
    if not pred_dir.exists():
        pred_dir = wandb_dir / "predictions"

    epoch_dirs = sorted(pred_dir.glob("epoch_*"))
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* dirs in {pred_dir}")

    if epoch is not None:
        epoch_dirs = [
            d for d in epoch_dirs if d.name == f"epoch_{epoch:04d}"
        ]
    epoch_dir = epoch_dirs[-1]
    files = sorted(epoch_dir.glob("*.parquet"))

    cols = ["refl_ids", "qi_mean"]
    df = pl.scan_parquet(files).select(cols).collect()
    return df, epoch_dir.name


def compute_consistency(qi_mean, H, K, L, min_mult):
    """Compute per-HKL consistency metrics."""
    df = pl.DataFrame({
        "H": H, "K": K, "L": L, "qi_mean": qi_mean,
    })

    stats = df.group_by(["H", "K", "L"]).agg([
        pl.col("qi_mean").count().alias("mult"),
        pl.col("qi_mean").mean().alias("mean_I"),
        pl.col("qi_mean").std().alias("std_I"),
        pl.col("qi_mean").median().alias("median_I"),
        (pl.col("qi_mean") - pl.col("qi_mean").mean()).abs().mean().alias("mad_from_mean"),
    ]).filter(pl.col("mult") >= min_mult)

    cv = (stats["std_I"] / stats["mean_I"].clip(lower_bound=1e-6)).to_numpy()
    mad_rel = (stats["mad_from_mean"] / stats["median_I"].clip(lower_bound=1e-6)).to_numpy()
    mult = stats["mult"].to_numpy()
    mean_I = stats["mean_I"].to_numpy()

    # R_merge
    merged = df.join(
        stats.select(["H", "K", "L", "mean_I"]),
        on=["H", "K", "L"],
    )
    numer = (merged["qi_mean"] - merged["mean_I"]).abs().sum()
    denom = merged["qi_mean"].abs().sum()
    r_merge = float(numer / max(denom, 1e-6))

    valid = np.isfinite(cv) & np.isfinite(mad_rel)
    return {
        "cv": cv[valid],
        "mad_rel": mad_rel[valid],
        "mult": mult[valid],
        "mean_I": mean_I[valid],
        "r_merge": r_merge,
        "n_hkl": int(valid.sum()),
    }


def main():
    import torch

    args = parse_args()
    meta = torch.load(args.metadata, weights_only=False)

    H = meta["H"].numpy().astype(np.int32)
    K = meta["K"].numpy().astype(np.int32)
    L = meta["L"].numpy().astype(np.int32)

    labels = args.labels or [d.name for d in args.run_dir]
    results = []

    for run_dir, label in zip(args.run_dir, labels):
        df, epoch_label = load_predictions(run_dir, args.epoch)
        refl_ids = df["refl_ids"].to_numpy().astype(np.int64)
        qi = df["qi_mean"].to_numpy().astype(np.float64)

        stats = compute_consistency(
            qi, H[refl_ids], K[refl_ids], L[refl_ids], args.min_mult
        )
        stats["label"] = label
        stats["epoch"] = epoch_label
        results.append(stats)

        print(f"\n{label} ({epoch_label}):")
        print(f"  HKLs with mult >= {args.min_mult}: {stats['n_hkl']}")
        print(f"  R_merge: {stats['r_merge']:.4f}")
        print(f"  Median CV: {np.median(stats['cv']):.4f}")
        print(f"  Median MAD/median: {np.median(stats['mad_rel']):.4f}")

    out_dir = args.out or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_runs = len(results)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_runs, 2)))

    # --- CV histogram ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        ax.hist(
            r["cv"].clip(0, 2), bins=100, alpha=0.5,
            label=f"{r['label']} (med={np.median(r['cv']):.3f})",
            color=colors[i], density=True,
        )
    ax.set_xlabel("CV (std / mean) per HKL")
    ax.set_ylabel("Density")
    ax.set_title(f"Per-HKL intensity consistency (mult ≥ {args.min_mult})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hkl_consistency_cv.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_dir / 'hkl_consistency_cv.png'}")
    plt.close(fig)

    # --- CV vs mean intensity ---
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), squeeze=False)
    for i, r in enumerate(results):
        ax = axes[0, i]
        mask = r["mean_I"] > 0
        ax.hexbin(
            np.log10(r["mean_I"][mask]),
            r["cv"][mask].clip(0, 2),
            gridsize=40, mincnt=1, cmap="viridis",
        )
        ax.set_xlabel("log₁₀(mean I)")
        ax.set_ylabel("CV")
        ax.set_title(f"{r['label']}\nR_merge={r['r_merge']:.4f}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"CV vs intensity (mult ≥ {args.min_mult})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "hkl_consistency_cv_vs_I.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'hkl_consistency_cv_vs_I.png'}")
    plt.close(fig)

    # --- CV vs multiplicity ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(results):
        mult_bins = np.arange(args.min_mult, min(r["mult"].max() + 1, 50))
        median_cv = []
        for m in mult_bins:
            sel = r["mult"] == m
            if sel.sum() > 10:
                median_cv.append(np.median(r["cv"][sel]))
            else:
                median_cv.append(np.nan)
        ax.plot(
            mult_bins, median_cv, "o-", color=colors[i], markersize=3,
            label=r["label"],
        )
    ax.set_xlabel("Multiplicity")
    ax.set_ylabel("Median CV")
    ax.set_title("Consistency vs redundancy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "hkl_consistency_vs_mult.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out_dir / 'hkl_consistency_vs_mult.png'}")
    plt.close(fig)

    # --- Summary table ---
    print(f"\n{'Label':<30} {'R_merge':>8} {'Med CV':>8} {'Med MAD/med':>12} {'N_HKL':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['label']:<30} {r['r_merge']:>8.4f} "
            f"{np.median(r['cv']):>8.4f} {np.median(r['mad_rel']):>12.4f} "
            f"{r['n_hkl']:>8d}"
        )


if __name__ == "__main__":
    main()
