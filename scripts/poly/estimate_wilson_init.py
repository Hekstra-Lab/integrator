"""Estimate the Wilson prior's scale and B-factor from the raw counts.

The Wilson prior is `<I> = G * exp(-B / (2 d^2))`, so `G` is the expected
intensity at low resolution *in the data's own count units*. The default
`G = 1` therefore states that a reflection is worth about one photon, which
for real data is orders of magnitude low and drags q(I) down for the first
epochs of training.

This reads a random sample of shoeboxes straight off disk (memory-mapped,
so it touches only the sampled rows) and runs the same binned Wilson-plot
fit the loss uses, printing the config lines to paste.

Usage:
    python scripts/poly/estimate_wilson_init.py --data-dir <dataset dir>
    python scripts/poly/estimate_wilson_init.py --data-dir <dir> --n 50000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description="Wilson G/B init from raw counts")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument(
        "--n", type=int, default=20000, help="Shoeboxes to sample"
    )
    p.add_argument("--bins", type=int, default=20, help="Resolution bins")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    from integrator.io import load_metadata, read_dataset_spec
    from integrator.model.loss.wilson_loss import WilsonLoss

    spec = read_dataset_spec(args.data_dir)
    if spec is None:
        raise SystemExit(f"no dataset.yaml in {args.data_dir}")
    files = spec.get("files", {})
    geom = spec["geometry"]
    n_px = geom["d"] * geom["h"] * geom["w"]

    counts = np.load(
        args.data_dir / files.get("counts", "counts.npy"), mmap_mode="r"
    )
    masks = np.load(
        args.data_dir / files.get("masks", "masks.npy"), mmap_mode="r"
    )
    meta = load_metadata(args.data_dir / files.get("reference", "metadata.npy"))
    d_all = np.asarray(meta["d"], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    n = min(args.n, counts.shape[0])
    idx = np.sort(rng.choice(counts.shape[0], size=n, replace=False))

    # memory-mapped fancy indexing pulls only the sampled rows
    c = np.asarray(counts[idx], dtype=np.float64).reshape(n, -1)
    m = np.asarray(masks[idx], dtype=np.float64).reshape(n, -1)
    d = d_all[idx]

    # per-shoebox background from the median masked pixel: robust to the peak
    masked = np.where(m > 0, c, np.nan)
    bg = np.nanmedian(masked, axis=-1)
    i_hat = np.nansum((masked - bg[:, None]) * (m > 0), axis=-1)

    good = np.isfinite(i_hat) & np.isfinite(d) & (d > 0)
    i_hat, d = i_hat[good], d[good]
    s_sq = 1.0 / (4.0 * d**2)

    g0, b0 = WilsonLoss.wilson_fit(
        torch.from_numpy(i_hat), torch.from_numpy(s_sq), args.bins
    )

    print(f"\nsampled {good.sum():,} of {counts.shape[0]:,} reflections")
    print(f"  pixels per shoebox    {n_px}")
    print(f"  median background     {np.median(bg):.2f} counts/pixel")
    print(f"  median I (bg-subtr.)  {np.median(i_hat):,.0f} counts")
    print(f"  resolution range      {d.min():.2f} – {d.max():.2f} A")
    print("\nWilson-plot fit:")
    print(f"  G = {g0:,.1f}   (prior <I> at low resolution, in counts)")
    print(f"  B = {b0:.1f} A^2")
    print("\nconfig:")
    print("  loss:\n    args:")
    print(f"      init_G: {g0:.1f}")
    print(f"      init_B: {max(b0, 0.0):.1f}")
    if g0 < 10:
        print(
            "\nnote: G < 10 suggests the background subtraction ate the "
            "signal; check the median background above."
        )


if __name__ == "__main__":
    main()
