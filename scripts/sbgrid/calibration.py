"""Is the posterior calibrated, or is it over-shrinking?

sd/mean of the posterior means cannot answer this. The Wilson value of 1.0
describes the TRUE intensities; posterior means are shrunk estimates and are
*supposed* to have lower dispersion, so a low sd/mean is consistent with a
good prior and a bad one alike.

What separates them is calibration. With ~50-fold multiplicity, the other
observations of the same reflection give a low-noise reference for the one
being tested, so each posterior can be scored against a near-truth:

    z = (E_q[I] - I_ref * s) / sqrt(Var_q[I] + Var[I_ref * s])

Well calibrated means sd(z) ~ 1 and central 68/95% intervals covering 68/95%
of the time. sd(z) > 1 is overconfidence -- the posterior is too narrow, or
its mean is biased, which is what over-shrinkage would look like. sd(z) < 1
means the error bars are too wide.

The reference is an UNWEIGHTED leave-one-out mean. Weighting by 1/sigma^2
biases weak reflections low when the weights correlate with the value, which
is exactly the regime under test.

Usage:
    python scripts/sbgrid/calibration.py --scaled <arm>/scaled.refl
    python scripts/sbgrid/calibration.py --scaled a/scaled.refl --label ours \
        --scaled-b b/scaled.refl --label-b DIALS
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

SHELLS = ((99, 3.0), (3.0, 2.4), (2.4, 2.1), (2.1, 1.96),
          (1.96, 1.87), (1.87, 1.79), (1.79, 1.70))


def parse_args():
    p = argparse.ArgumentParser(description="Posterior calibration by shell")
    p.add_argument("--scaled", required=True, help="scaled.refl for the arm")
    p.add_argument("--label", default="arm")
    p.add_argument("--scaled-b", default=None, help="a second arm to compare")
    p.add_argument("--label-b", default="arm B")
    p.add_argument(
        "--intensity",
        default="prf",
        choices=("prf", "sum"),
        help="which column carries the estimate under test",
    )
    p.add_argument("--min-multiplicity", type=int, default=8)
    return p.parse_args()


def load(path: str, which: str):
    from dials.array_family import flex

    table = flex.reflection_table.from_file(path)
    value = np.asarray(table[f"intensity.{which}.value"], dtype=np.float64)
    variance = np.asarray(table[f"intensity.{which}.variance"], dtype=np.float64)
    scale = np.asarray(table["inverse_scale_factor"], dtype=np.float64)
    d = np.asarray(table["d"], dtype=np.float64)
    hkl = np.asarray(table["miller_index"]).reshape(-1, 3)
    keep = np.isfinite(value) & np.isfinite(variance) & (variance > 0) & (scale > 0)
    return value[keep], variance[keep], scale[keep], d[keep], hkl[keep]


def leave_one_out(value, scale, groups):
    """Per observation: the unweighted mean and variance of its siblings.

    Computed by group sums minus the observation itself, so it is one pass
    rather than one pass per reflection.
    """
    scaled = value / scale
    n = np.bincount(groups)
    total = np.bincount(groups, weights=scaled)
    total_sq = np.bincount(groups, weights=scaled**2)

    count = n[groups] - 1
    mean = (total[groups] - scaled) / np.maximum(count, 1)
    # unbiased variance of the siblings, then of their mean
    sq = (total_sq[groups] - scaled**2) / np.maximum(count, 1)
    var = np.maximum(sq - mean**2, 0) * count / np.maximum(count - 1, 1)
    return mean, var / np.maximum(count, 1), count


def report(label, value, variance, scale, d, hkl, min_multiplicity):
    _, index = np.unique(hkl, axis=0, return_inverse=True)
    reference, reference_var, count = leave_one_out(value, scale, index)

    predicted = reference * scale
    predicted_var = reference_var * scale**2
    z = (value - predicted) / np.sqrt(variance + predicted_var)

    usable = (count >= min_multiplicity) & np.isfinite(z)
    print(f"\n{label}: {usable.sum():,} observations with multiplicity "
          f">= {min_multiplicity} (median {np.median(count[usable]):.0f})")
    print(f"  {'shell (A)':>13s}{'sd(z)':>9s}{'mean(z)':>10s}"
          f"{'|z|<1':>8s}{'|z|<1.96':>10s}{'n':>9s}")
    for hi, lo in SHELLS:
        m = usable & (d <= hi) & (d > lo)
        if m.sum() < 200:
            continue
        zz = z[m]
        print(f"  {f'{hi:.2f}-{lo:.2f}':>13s}{zz.std():>9.2f}{zz.mean():>10.3f}"
              f"{np.mean(np.abs(zz) < 1) * 100:>7.0f}%"
              f"{np.mean(np.abs(zz) < 1.96) * 100:>9.0f}%{m.sum():>9d}")
    print("  ideal            1.00     0.000     68%       95%")


def main():
    args = parse_args()
    report(args.label, *load(args.scaled, args.intensity), args.min_multiplicity)
    if args.scaled_b:
        report(args.label_b, *load(args.scaled_b, args.intensity),
               args.min_multiplicity)
    print("\n  sd(z) > 1: posterior too narrow or its mean biased "
          "(over-shrinkage looks like this)")
    print("  sd(z) < 1: error bars too wide")
    print("  mean(z) away from 0: systematic bias against the siblings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
