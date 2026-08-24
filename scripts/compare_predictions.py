"""Compare reflection positions between two MTZ files (e.g. laue-dials vs precognition).

Plots x,y positions for a single image, colored by intensity.

Usage:
    uv run python scripts/compare_predictions.py \
        --mtz1 laue_dials.mtz --label1 "laue-dials" \
        --mtz2 precognition.mtz --label2 "precognition" \
        --image 100 \
        [--out comparison.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare reflection positions from two MTZ files"
    )
    parser.add_argument("--mtz1", type=Path, required=True)
    parser.add_argument("--label1", type=str, default="mtz1")
    parser.add_argument("--mtz2", type=Path, required=True)
    parser.add_argument("--label2", type=str, default="mtz2")
    parser.add_argument("--image", type=int, default=None,
                        help="Image number to plot (default: first available)")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_mtz(path):
    import reciprocalspaceship as rs
    ds = rs.read_mtz(str(path))
    print(f"Loaded {path.name}: {len(ds)} reflections")
    print(f"  Columns: {list(ds.columns)}")
    # Find position columns
    x_col = None
    y_col = None
    batch_col = None
    I_col = None
    for c in ds.columns:
        cl = c.lower()
        if cl in ("xcal", "x", "xdet", "xcal.px", "xyzcal.px.0"):
            x_col = c
        if cl in ("ycal", "y", "ydet", "ycal.px", "xyzcal.px.1"):
            y_col = c
        if cl in ("batch", "image", "image_num", "img"):
            batch_col = c
        if cl in ("i", "iobs", "intensity", "intensity.sum.value"):
            I_col = c
    print(f"  x={x_col}, y={y_col}, batch={batch_col}, I={I_col}")
    return ds, x_col, y_col, batch_col, I_col


def main():
    args = parse_args()

    ds1, x1, y1, b1, I1 = load_mtz(args.mtz1)
    ds2, x2, y2, b2, I2 = load_mtz(args.mtz2)

    # Determine image to plot
    if args.image is not None:
        img = args.image
    else:
        if b1 is not None:
            img = int(ds1[b1].min())
        elif b2 is not None:
            img = int(ds2[b2].min())
        else:
            img = None

    # Filter to image
    if img is not None and b1 is not None:
        mask1 = ds1[b1].to_numpy().astype(int) == img
        d1 = ds1[mask1]
    else:
        d1 = ds1
        img = "all"

    if img != "all" and b2 is not None:
        mask2 = ds2[b2].to_numpy().astype(int) == img
        d2 = ds2[mask2]
    else:
        d2 = ds2

    print(f"\nImage {img}: {len(d1)} ({args.label1}), {len(d2)} ({args.label2})")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: mtz1
    ax = axes[0]
    if x1 and y1:
        xx1 = d1[x1].to_numpy().astype(float)
        yy1 = d1[y1].to_numpy().astype(float)
        if I1:
            cc1 = d1[I1].to_numpy().astype(float)
            sc = ax.scatter(xx1, yy1, c=cc1, s=2, alpha=0.5, cmap="viridis",
                           norm=plt.matplotlib.colors.SymLogNorm(linthresh=1))
            fig.colorbar(sc, ax=ax, label=I1, shrink=0.8)
        else:
            ax.scatter(xx1, yy1, s=2, alpha=0.5)
    ax.set_title(f"{args.label1} (N={len(d1)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Panel 2: mtz2
    ax = axes[1]
    if x2 and y2:
        xx2 = d2[x2].to_numpy().astype(float)
        yy2 = d2[y2].to_numpy().astype(float)
        if I2:
            cc2 = d2[I2].to_numpy().astype(float)
            sc = ax.scatter(xx2, yy2, c=cc2, s=2, alpha=0.5, cmap="viridis",
                           norm=plt.matplotlib.colors.SymLogNorm(linthresh=1))
            fig.colorbar(sc, ax=ax, label=I2, shrink=0.8)
        else:
            ax.scatter(xx2, yy2, s=2, alpha=0.5)
    ax.set_title(f"{args.label2} (N={len(d2)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Panel 3: overlay
    ax = axes[2]
    if x1 and y1:
        ax.scatter(xx1, yy1, s=3, alpha=0.4, label=args.label1, c="blue")
    if x2 and y2:
        ax.scatter(xx2, yy2, s=3, alpha=0.4, label=args.label2, c="red")
    ax.set_title(f"Overlay - image {img}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(markerscale=5)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Reflection positions - image {img}", fontsize=12)
    fig.tight_layout()

    out = args.out or f"compare_positions_img{img}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
