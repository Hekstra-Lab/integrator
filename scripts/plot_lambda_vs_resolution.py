"""Hexbin plot of the (wavelength, 1/d²) plane for a Laue dataset.

Two side-by-side panels:
    1) colored by mean(intensity) - bright hexes show where the
       strong reflections concentrate in the (λ, d) plane
    2) colored by count - shows how the model's wavelength bins and
       resolution shells will be populated

A quick visual check that the dataset actually spans the wavelength range
you assume in your Wilson hyperprior, and that the joint (λ, d²) coverage
isn't too sparse anywhere.

Usage (metadata.pt):
    uv run python scripts/plot_lambda_vs_resolution.py \
        --metadata /n/.../pytorch_data/metadata.pt \
        --out /n/.../pytorch_data/lambda_vs_resolution.png \
        --gridsize 60

Usage (directory of parquets):
    uv run python scripts/plot_lambda_vs_resolution.py \
        --parquet-dir /n/.../predictions/ \
        --out /n/.../lambda_vs_resolution.png \
        --gridsize 60
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


def parse_args():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--metadata",
        type=str,
        help="path to metadata.pt produced by refltorch.mksbox-laue",
    )
    source.add_argument(
        "--parquet-dir",
        type=str,
        help="directory containing parquet files with qi_mean, wavelength, d columns",
    )
    p.add_argument(
        "--out",
        type=str,
        default="lambda_vs_resolution.png",
        help="output figure path",
    )
    p.add_argument(
        "--gridsize",
        type=int,
        default=60,
        help="hexbin grid size",
    )
    p.add_argument(
        "--intensity-key",
        type=str,
        default="intensity.sum.value",
        help="key in metadata.pt to use for intensity coloring",
    )
    p.add_argument(
        "--clip-percentile",
        type=float,
        default=99.5,
        help="clip the intensity colorbar at this percentile (robust to outliers)",
    )
    return p.parse_args()


def load_from_metadata(args):
    meta = torch.load(args.metadata, weights_only=True, map_location="cpu")
    for required in ("d", "wavelength", args.intensity_key):
        if required not in meta:
            raise KeyError(
                f"metadata.pt missing required key '{required}'. Has: "
                f"{sorted(meta.keys())}"
            )
    d = meta["d"].numpy().astype(np.float64)
    lam = meta["wavelength"].numpy().astype(np.float64)
    I = meta[args.intensity_key].numpy().astype(np.float64)
    label = Path(args.metadata).parent.name
    return d, lam, I, label


def load_from_parquets(args):
    import polars as pl

    parquet_dir = Path(args.parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    df = pl.scan_parquet(files).select("qi_mean", "wavelength", "d").collect()
    d = df["d"].to_numpy().astype(np.float64)
    lam = df["wavelength"].to_numpy().astype(np.float64)
    I = df["qi_mean"].to_numpy().astype(np.float64)
    label = parquet_dir.name
    return d, lam, I, label


def main():
    args = parse_args()

    if args.parquet_dir is not None:
        d, lam, I, label = load_from_parquets(args)
        intensity_label = "mean(qi_mean)"
    else:
        d, lam, I, label = load_from_metadata(args)
        intensity_label = f"mean({args.intensity_key})"

    valid = np.isfinite(d) & np.isfinite(lam) & np.isfinite(I) & (d > 0)
    d, lam, I = d[valid], lam[valid], I[valid]

    s_sq = 1.0 / (4.0 * d**2)

    I_lo = max(np.percentile(I, 1.0), 1.0)
    I_hi = np.percentile(I, args.clip_percentile)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    ax = axes[0]
    hb1 = ax.hexbin(
        lam,
        s_sq,
        C=np.clip(I, I_lo, I_hi),
        reduce_C_function=np.mean,
        gridsize=args.gridsize,
        mincnt=1,
        cmap="viridis",
        norm=LogNorm(vmin=I_lo, vmax=I_hi),
    )
    cb1 = fig.colorbar(hb1, ax=ax)
    cb1.set_label(intensity_label)
    ax.set_xlabel("wavelength λ (Å)")
    ax.set_ylabel("1 / (4 d²)  (Å⁻²)")
    ax.set_title("colored by mean intensity")

    ax = axes[1]
    hb2 = ax.hexbin(
        lam,
        s_sq,
        gridsize=args.gridsize,
        mincnt=1,
        cmap="magma",
        norm=LogNorm(),
    )
    cb2 = fig.colorbar(hb2, ax=ax)
    cb2.set_label("# reflections")
    ax.set_xlabel("wavelength λ (Å)")
    ax.set_title("colored by number of reflections")

    fig.suptitle(
        f"{label}  "
        f"({len(d)} refls,  λ ∈ [{lam.min():.3f}, {lam.max():.3f}] Å,  "
        f"d ∈ [{d.min():.2f}, {d.max():.2f}] Å)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
