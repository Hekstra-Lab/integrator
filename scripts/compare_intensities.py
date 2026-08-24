"""Compare predicted intensities (qi_mean) vs DIALS intensities.

Usage:
    uv run python scripts/compare_intensities.py <run_dir> \
        [--epoch EPOCH] [--out intensities.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare predicted vs DIALS intensities"
    )
    parser.add_argument(
        "run_dir", type=Path, help="Run directory with run_paths.yaml"
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch to visualize (default: latest)",
    )
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_run_info(run_dir: Path) -> Path:
    meta = yaml.safe_load((run_dir / "run_paths.yaml").read_text())
    log_dir = Path(meta["wandb"]["log_dir"])
    return log_dir.parent / "predictions"


def find_epoch_dir(pred_dir: Path, epoch: int | None) -> Path:
    epoch_dirs = sorted(d for d in pred_dir.glob("epoch_*") if d.is_dir())
    if not epoch_dirs:
        raise FileNotFoundError(f"No epoch_* dirs in {pred_dir}")
    if epoch is not None:
        target = pred_dir / f"epoch_{epoch:04d}"
        if not target.exists():
            raise FileNotFoundError(f"{target} not found")
        return target
    return epoch_dirs[-1]


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()

    pred_dir = load_run_info(run_dir)
    epoch_dir = find_epoch_dir(pred_dir, args.epoch)
    epoch_num = epoch_dir.name.replace("epoch_", "")

    parquets = sorted(epoch_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets in {epoch_dir}")

    df = pl.read_parquet(parquets)

    if "qi_mean" not in df.columns or "intensity.sum.value" not in df.columns:
        raise ValueError("Need qi_mean and intensity.sum.value in parquets")

    qi = df["qi_mean"].to_numpy()
    dials = df["intensity.sum.value"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(qi, dials, s=0.5, alpha=0.05, c="steelblue", edgecolors="none")

    # x=y line
    lims = [
        min(np.percentile(qi, 0.1), np.percentile(dials, 0.1), -10),
        max(np.percentile(qi, 99.9), np.percentile(dials, 99.9)),
    ]
    ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7, label="x = y")

    ax.set_xlabel("Predicted qi_mean", fontsize=11)
    ax.set_ylabel("DIALS intensity.sum.value", fontsize=11)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Stats
    valid = np.isfinite(qi) & np.isfinite(dials)
    corr = np.corrcoef(qi[valid], dials[valid])[0, 1]
    ratio = np.median(qi[valid] / np.clip(dials[valid], 1, None))

    ax.set_title(
        f"Epoch {epoch_num}  |  N={valid.sum():,}  |  "
        f"corr={corr:.3f}  |  median(qi/dials)={ratio:.2f}",
        fontsize=10,
    )

    plt.tight_layout()
    out = args.out or f"intensity_comparison_epoch{epoch_num}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
