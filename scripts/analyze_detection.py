"""Analyze zero-inflated model detection probabilities.

Usage:
    uv run python scripts/analyze_detection.py <run_dir> [--epoch EPOCH]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze ZI model detection probabilities"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_predictions(run_dir, epoch):
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    pred_dir = Path(meta["wandb"]["log_dir"]).parent / "predictions"
    epoch_dirs = sorted(d for d in pred_dir.glob("epoch_*") if d.is_dir())
    if epoch is not None:
        epoch_dir = pred_dir / f"epoch_{epoch:04d}"
    else:
        epoch_dir = epoch_dirs[-1]
    parquets = sorted(epoch_dir.glob("*.parquet"))
    df = pl.read_parquet(parquets)
    return df, epoch_dir.name.replace("epoch_", "")


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    df, epoch_num = load_predictions(run_dir, args.epoch)

    if "qi_pi" not in df.columns:
        print("No qi_pi column - not a zero-inflated model.")
        return

    pi = df["qi_pi"].to_numpy()
    qi = df["qi_mean"].to_numpy()
    dials = df["intensity.sum.value"].to_numpy() if "intensity.sum.value" in df.columns else None

    # 1. Pi distribution
    print(f"=== Detection probability π (epoch {epoch_num}) ===")
    print(f"  N reflections: {len(pi):,}")
    print(f"  mean: {pi.mean():.3f}")
    print(f"  std:  {pi.std():.3f}")
    print(f"  min:  {pi.min():.4f}")
    print(f"  max:  {pi.max():.4f}")
    print()
    print("  Distribution:")
    for edge in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]:
        print(f"    π < {edge:.2f}: {(pi < edge).mean()*100:.1f}%")
    print()

    # 2. Pi vs DIALS intensity
    if dials is not None:
        print("=== π vs DIALS intensity ===")
        for label, mask in [
            ("dials < 0", dials < 0),
            ("dials 0-10", (dials >= 0) & (dials < 10)),
            ("dials 10-100", (dials >= 10) & (dials < 100)),
            ("dials 100-1000", (dials >= 100) & (dials < 1000)),
            ("dials > 1000", dials >= 1000),
        ]:
            if mask.sum() > 0:
                print(
                    f"  {label:>16s}: N={mask.sum():>8,}  "
                    f"π_mean={pi[mask].mean():.3f}  "
                    f"π_median={np.median(pi[mask]):.3f}  "
                    f"qi_mean={qi[mask].mean():.1f}"
                )
        print()

    # 3. Weak vs strong
    print("=== Weak vs strong reflections ===")
    weak = qi < 10
    strong = qi > 100
    print(f"  Weak (qi<10):   N={weak.sum():>8,}  π_mean={pi[weak].mean():.3f}  π_median={np.median(pi[weak]):.3f}")
    print(f"  Strong (qi>100): N={strong.sum():>8,}  π_mean={pi[strong].mean():.3f}  π_median={np.median(pi[strong]):.3f}")
    print()

    # 4. qi percentiles
    print("=== qi_mean percentiles ===")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
        print(f"  {p:>3d}%: {np.percentile(qi, p):.2f}")
    print()

    # 5. Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Pi histogram
    ax = axes[0, 0]
    ax.hist(pi, bins=100, edgecolor="none", alpha=0.8)
    ax.set_xlabel("π (detection probability)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of π")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5)

    # Pi vs qi
    ax = axes[0, 1]
    ax.scatter(pi, qi, s=0.3, alpha=0.02, edgecolors="none")
    ax.set_xlabel("π")
    ax.set_ylabel("qi_mean")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_title("π vs qi_mean")
    ax.grid(alpha=0.3)

    # Pi vs DIALS
    ax = axes[1, 0]
    if dials is not None:
        ax.scatter(pi, dials, s=0.3, alpha=0.02, edgecolors="none")
        ax.set_xlabel("π")
        ax.set_ylabel("DIALS intensity")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title("π vs DIALS intensity")
        ax.grid(alpha=0.3)
    else:
        ax.set_visible(False)

    # qi vs DIALS colored by pi
    ax = axes[1, 1]
    if dials is not None:
        sc = ax.scatter(qi, dials, c=pi, s=0.5, alpha=0.1, cmap="coolwarm",
                        vmin=0, vmax=1, edgecolors="none")
        ax.plot([0.1, qi.max()], [0.1, qi.max()], "k--", alpha=0.3)
        ax.set_xlabel("qi_mean")
        ax.set_ylabel("DIALS intensity")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title("qi vs DIALS (colored by π)")
        ax.grid(alpha=0.3)
        fig.colorbar(sc, ax=ax, label="π", shrink=0.8)
    else:
        ax.set_visible(False)

    fig.suptitle(f"Zero-inflated detection analysis - epoch {epoch_num}", fontsize=12)
    fig.tight_layout()

    out = args.out or f"detection_analysis_epoch{epoch_num}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
