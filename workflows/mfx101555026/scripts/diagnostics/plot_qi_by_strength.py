#!/usr/bin/env python3
"""
plot_qi_by_strength.py — qi distribution stratified by reflection strength.

Bins reflections into weak / medium / strong using cctbx intensity percentiles
computed from intensity.sum.value:
  weak   : I < p10   (bottom 10th percentile)
  medium : p10 ≤ I < p90
  strong : I ≥ p90   (top 10th percentile)

The p10/p50/p90 boundaries are derived from the cctbx column
intensity.sum.value written into the prediction parquet by integrator.predict.

Usage
-----
python plot_qi_by_strength.py \\
    --run-dir  $RUN_DIR \\
    --epoch    24 \\
    --out-dir  /tmp/qi_by_strength \\
    [--p-weak 10] [--p-strong 90] [--n-profile 625] [--dpi 150]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot qi distribution stratified by weak/medium/strong "
            "using cctbx intensity percentiles (p10/p50/p90)."
        ),
    )
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="integrator run directory (contains files/predictions/)",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch index (e.g. 24). Defaults to the latest epoch found.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files. Defaults to <run-dir>/diagnostics/",
    )
    p.add_argument(
        "--p-weak",
        type=int,
        default=10,
        help="Percentile upper bound for 'weak' category (default: 10 → p10)",
    )
    p.add_argument(
        "--p-strong",
        type=int,
        default=90,
        help="Percentile lower bound for 'strong' category (default: 90 → p90)",
    )
    p.add_argument(
        "--n-profile",
        type=int,
        default=625,
        help="Number of pixels in the flattened profile (default: 625 = 25×25)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Max rows to sample per category for speed (default: 500 000)",
    )
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_pred_dir(run_dir: Path, epoch: int | None) -> Path:
    pred_root = run_dir / "files" / "predictions"
    if not pred_root.exists():
        raise FileNotFoundError(
            f"No predictions directory found at {pred_root}"
        )
    if epoch is not None:
        d = pred_root / f"epoch_{epoch:04d}"
        if not d.exists():
            raise FileNotFoundError(f"Epoch directory not found: {d}")
        return d
    # find latest epoch_XXXX directory
    epoch_dirs = sorted(pred_root.glob("epoch_*"))
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* directories found in {pred_root}")
    return epoch_dirs[-1]


def _load_predictions(pred_dir: Path, max_rows: int) -> pl.DataFrame:
    parquet_files = sorted(pred_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {pred_dir}")
    logger.info(
        "Scanning %d parquet files from %s", len(parquet_files), pred_dir
    )

    needed = [
        "qi_mean",
        "qi_var",
        "intensity.sum.value",
        "intensity.sum.variance",
        "d",
    ]
    # also load qp_mean columns if present (flattened profile)
    lf = pl.scan_parquet(parquet_files)
    schema_cols = lf.collect_schema().names()
    qp_cols = [c for c in schema_cols if c.startswith("qp_mean")]
    select_cols = [c for c in needed if c in schema_cols] + qp_cols

    df = lf.select(select_cols).collect()
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if len(df) > max_rows:
        logger.info("Sampling %d rows from %d total", max_rows, len(df))
        df = df.sample(max_rows, seed=42)

    return df


def _bin_by_strength(
    df: pl.DataFrame,
    p_weak: int,
    p_strong: int,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Bin by cctbx intensity percentiles.

    Uses intensity.sum.value directly (cctbx column).
    Returns (bins dict, p_weak boundary, p_strong boundary).
    """
    I = df["intensity.sum.value"].to_numpy().astype(np.float64)
    lo = float(np.percentile(I, p_weak))
    hi = float(np.percentile(I, p_strong))
    bins = {
        f"weak  (I < p{p_weak} = {lo:.1f})": np.where(I < lo)[0],
        f"medium (p{p_weak} ≤ I < p{p_strong})": np.where(
            (I >= lo) & (I < hi)
        )[0],
        f"strong (I ≥ p{p_strong} = {hi:.1f})": np.where(I >= hi)[0],
    }
    return bins, lo, hi


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_qi_distributions(
    df: pl.DataFrame,
    bins: dict[str, np.ndarray],
    out_dir: Path,
    dpi: int,
) -> None:
    """Histogram of qi_mean for each strength category on the same axes."""
    qi = df["qi_mean"].to_numpy().astype(np.float32)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "darkorange", "seagreen"]

    for (label, idx), color in zip(bins.items(), colors):
        if len(idx) == 0:
            logger.warning("No reflections in category: %s", label)
            continue
        vals = qi[idx]
        ax.hist(
            vals,
            bins=100,
            range=(0, np.percentile(qi, 99)),
            density=True,
            alpha=0.55,
            label=f"{label}  (n={len(idx):,})",
            color=color,
        )

    ax.set_xlabel("qi_mean (predicted intensity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "qi distribution by reflection strength (cctbx intensity percentiles)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = out_dir / "qi_by_strength.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_mean_profiles(
    df: pl.DataFrame,
    bins: dict[str, np.ndarray],
    n_profile: int,
    out_dir: Path,
    dpi: int,
) -> None:
    """Mean learned spot profile (qp_mean) for each strength category."""
    qp_cols = [c for c in df.columns if c.startswith("qp_mean")]
    if len(qp_cols) != n_profile:
        logger.warning(
            "Expected %d qp_mean columns but found %d — skipping profile plot.",
            n_profile,
            len(qp_cols),
        )
        return

    qp = df.select(qp_cols).to_numpy().astype(np.float32)  # (N, pixels)
    h = w = int(np.sqrt(n_profile))
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (label, idx), color in zip(axes, bins.items(), colors):
        if len(idx) == 0:
            ax.set_title(f"{label}\n(no data)")
            ax.axis("off")
            continue
        mean_profile = qp[idx].mean(axis=0).reshape(h, w)
        im = ax.imshow(mean_profile, cmap="inferno", origin="lower")
        ax.set_title(f"{label}\nn={len(idx):,}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Mean learned spot profile (qp_mean) by reflection strength",
        fontsize=12,
    )
    fig.tight_layout()
    out = out_dir / "mean_profile_by_strength.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", out)


def plot_qi_vs_d(
    df: pl.DataFrame,
    bins: dict[str, np.ndarray],
    out_dir: Path,
    dpi: int,
) -> None:
    """Scatter / 2D hex of qi_mean vs resolution d for each category."""
    if "d" not in df.columns:
        return
    qi = df["qi_mean"].to_numpy().astype(np.float32)
    d = df["d"].to_numpy().astype(np.float32)
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    q99 = np.percentile(qi, 99)
    for ax, (label, idx), color in zip(axes, bins.items(), colors):
        if len(idx) == 0:
            ax.set_title(f"{label}\n(no data)")
            continue
        sample = np.random.default_rng(0).choice(
            idx, min(len(idx), 50_000), replace=False
        )
        ax.hexbin(
            d[sample],
            qi[sample],
            gridsize=60,
            extent=(d.min(), d.max(), 0, q99),
            cmap="Blues",
            mincnt=1,
        )
        ax.set_xlabel("d (Å)", fontsize=10)
        ax.set_title(f"{label}\nn={len(idx):,}", fontsize=9)
    axes[0].set_ylabel("qi_mean", fontsize=10)
    fig.suptitle("qi_mean vs resolution by reflection strength", fontsize=12)
    fig.tight_layout()
    out = out_dir / "qi_vs_d_by_strength.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    out_dir = args.out_dir or (args.run_dir / "diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = _find_pred_dir(args.run_dir, args.epoch)
    logger.info("Prediction dir: %s", pred_dir)

    df = _load_predictions(pred_dir, args.max_rows)

    for col in ("intensity.sum.value", "qi_mean"):
        if col not in df.columns:
            raise RuntimeError(
                f"Required column '{col}' not found in predictions.\n"
                "Make sure the training YAML includes it in predict_keys."
            )

    bins, lo, hi = _bin_by_strength(df, args.p_weak, args.p_strong)
    logger.info(
        "Intensity percentile boundaries: p%d=%.2f  p%d=%.2f",
        args.p_weak,
        lo,
        args.p_strong,
        hi,
    )

    for label, idx in bins.items():
        logger.info("  %-50s  %8d reflections", label, len(idx))

    plot_qi_distributions(df, bins, out_dir, args.dpi)
    plot_mean_profiles(df, bins, args.n_profile, out_dir, args.dpi)
    plot_qi_vs_d(df, bins, out_dir, args.dpi)

    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
