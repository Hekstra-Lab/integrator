"""Re-merge the integrator's intensities under a different variance model.

The outer-shell CC-half deficit on 821 does not move when the Wilson prior
changes, so it is not the prior. What does change with resolution is how well
the posterior variance tracks the intensity: Spearman rho between I and sigma
falls from 0.999 to -0.18 by 1.70 A, while DIALS holds 0.73. Since
`dials.scale` and `dials.merge` weight by 1/sigma^2, weights out there carry
variation unrelated to precision.

This swaps the variance while holding the intensities fixed, so the merge sees
the same numbers with different weights. If CC-half recovers, the deficit is
in our error bars; if it does not, it is in the intensities themselves. Those
are different problems, and the merge cannot tell us which without this test.

    poisson  var = I + n_eff * bg, the model's own generative variance, with
             n_eff calibrated once against the posterior at low resolution
             where it is data-dominated (rho 0.999)
    dials    DIALS' own summation variance -- borrows information the model
             does not have, so it is a diagnostic ceiling, not a proposal
    model    the posterior variance, unchanged (the control)

Usage:
    python scripts/sbgrid/remerge_variance.py --pred-dir <epoch> \
        --dataset-dir <combined dataset> --variance poisson
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Re-merge under another variance")
    p.add_argument("--pred-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--variance", choices=("poisson", "dials", "model"), default="poisson"
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--calib-dmin",
        type=float,
        default=2.7,
        help="resolution above which the posterior is data-dominated, used to "
        "calibrate n_eff",
    )
    return p.parse_args()


def effective_pixels(frame: pl.DataFrame, dmin: float) -> float:
    """Pixels implied by the posterior variance where the data dominates.

    The generative model is counts ~ Poisson(I p + bg) per pixel, so a summed
    intensity has variance I + n bg for some effective count n. Rather than
    guess n from the shoebox size -- most of whose pixels carry no profile --
    it is read off the posterior itself in the low-resolution shells, where
    the variance is set by the data rather than the prior. That keeps the
    substitution on the model's own scale, so only the *shape* of the
    variance-intensity relationship changes.
    """
    low = frame.filter(pl.col("d") > dmin)
    excess = (low["qi_var"] - low["qi_mean"]).to_numpy()
    background = low["qbg_mean"].to_numpy()
    usable = (background > 0) & (excess > 0)
    if usable.sum() < 100:
        raise SystemExit("too few low-resolution reflections to calibrate n_eff")
    return float(np.median(excess[usable] / background[usable]))


def main():
    args = parse_args()
    out_dir = args.out_dir or args.pred_dir / f"per_sweep_{args.variance}"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load((args.dataset_dir / "dataset.yaml").read_text())
    sweeps = spec.get("sweeps")
    if not sweeps:
        raise SystemExit(f"{args.dataset_dir}/dataset.yaml lists no sweeps")

    single = args.pred_dir / "pred.parquet"
    files = [single] if single.exists() else sorted(
        args.pred_dir.glob("preds_epoch_*.parquet")
    )
    if not files:
        raise SystemExit(f"no prediction parquet under {args.pred_dir}")
    frame = pl.concat([pl.read_parquet(f) for f in files])

    if args.variance == "poisson":
        n_eff = effective_pixels(frame, args.calib_dmin)
        print(f"calibrated n_eff = {n_eff:.1f} pixels (d > {args.calib_dmin} A)")
        variance = (
            frame["qi_mean"].to_numpy().clip(0)
            + n_eff * frame["qbg_mean"].to_numpy().clip(0)
        )
    elif args.variance == "dials":
        variance = frame["intensity.sum.variance"].to_numpy()
    else:
        variance = frame["qi_var"].to_numpy()
    frame = frame.with_columns(pl.Series("use_var", variance))

    # how well the substituted variance tracks the intensity, which is the
    # property under test
    from scipy.stats import spearmanr

    print(f"{'shell (A)':>14s}{'rho(I, sigma)':>15s}{'n':>9s}")
    for hi, lo in ((99, 2.7), (1.96, 1.87), (1.87, 1.79), (1.79, 1.70)):
        shell = frame.filter((pl.col("d") <= hi) & (pl.col("d") > lo))
        if len(shell) < 200:
            continue
        rho = spearmanr(
            shell["qi_mean"].to_numpy(), np.sqrt(shell["use_var"].to_numpy().clip(0))
        ).statistic
        print(f"{f'{hi:.2f}-{lo:.2f}':>14s}{rho:>15.3f}{len(shell):>9d}")

    from integrator.io.pred_io import write_refl_with_predictions

    written = []
    for i, sweep in enumerate(sweeps):
        subset = frame.filter(pl.col("sweep_id") == i).sort("refl_ids")
        source = args.dataset_dir / f"reflections_{sweep}.refl"
        if not source.exists():
            print(f"  {sweep}: no table at {source.name}, skipping")
            continue
        target = out_dir / f"{sweep}.refl"
        write_refl_with_predictions(
            refl_file=source,
            out_file=target,
            refl_ids=subset["refl_ids"].to_numpy(),
            i_value=subset["qi_mean"].to_numpy(),
            i_variance=subset["use_var"].to_numpy(),
            bg_mean=subset["qbg_mean"].to_numpy(),
        )
        written.append(target)
        print(f"  {sweep}: {len(subset):,} -> {target.name}")

    if not written:
        raise SystemExit("no reflection tables written")
    print(f"\n{len(written)} table(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
