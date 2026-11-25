"""CRPS of every exported intensity against the simulation's true intensity, the proper score that rewards accuracy and calibration together.

For each row of the score_exports.sh joined tables (in the scoring domain), with
the export's intensity I and sigma SIGI and the true intensity y:

    Gaussian N(I, SIGI^2) for point estimates with a variance (i_rb, laue.integrate):
        CRPS = s (z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)),  z = (y - I) / s
    Gamma moment-matched to (I, SIGI^2) for posteriors (methods matching --posterior-pattern):
        CRPS = y (2 F_k(y) - 1) - k th (2 F_{k+1}(y) - 1) - th / B(1/2, k)

CRPS is the expected absolute error of the predictive distribution, in photons;
it is minimised only by an estimate that is both unbiased and calibrated, so it
replaces reading bias, relative sd and z_sd separately (whose rankings
disagree). Reported as the mean of CRPS / max(y, 1) over all rows and per bin
of y; the first table a method appears in is used.

Writes --out (long CSV: method, dataset, bin, n, crps_rel, crps_abs) and prints
the wide table, sorted by the all-reflection value.

usage: python crps_against_truth.py JOINED.csv [JOINED.csv ...] --out crps.csv
           [--dataset mos010_pc_default] [--posterior-pattern 'posterior|_lc$']
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy import special, stats

BINS = ((0, np.inf), (10, 30), (30, 100), (100, 300), (300, 1000), (1000, 3000), (3000, np.inf))


def crps_norm(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_gamma(y, m, s):
    k, th = (m / s) ** 2, s**2 / m
    F = lambda a: special.gammainc(a, np.maximum(y, 0) / th)  # noqa: E731
    return y * (2 * F(k) - 1) - k * th * (2 * F(k + 1) - 1) - th / special.beta(0.5, k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("joined", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="mos010_pc_default")
    ap.add_argument("--posterior-pattern", default="posterior|_lc$")
    a = ap.parse_args()
    seen, rows = set(), []
    for f in a.joined:
        j = pl.read_csv(f, infer_schema_length=10000, columns=["method", "dataset", "i_true", "in_domain", "I", "SIGI"])
        j = j.filter((pl.col("dataset") == a.dataset) & pl.col("in_domain").cast(pl.Boolean))
        for m in j["method"].unique().sort().to_list():
            if m in seen:
                continue
            seen.add(m)
            x = j.filter(method=m).drop_nulls(["I", "SIGI"]).filter(pl.col("SIGI") > 0)
            y, I, s = x["i_true"].to_numpy(), x["I"].to_numpy(), x["SIGI"].to_numpy()
            post = bool(pl.Series([m]).str.contains(a.posterior_pattern)[0])
            cr = crps_gamma(y, np.maximum(I, 1e-6), s) if post else crps_norm(y, I, s)
            for lo, hi in BINS:
                sel = (y >= lo) & (y < hi)
                name = "all" if (lo == 0 and hi == np.inf) else (f">{lo:g}" if hi == np.inf else f"{lo:g}-{hi:g}")
                rows.append(dict(method=m, dataset=a.dataset, bin=name, n=int(sel.sum()),
                                 crps_rel=float(np.mean(cr[sel] / np.maximum(y[sel], 1))) if sel.any() else np.nan,
                                 crps_abs=float(np.mean(cr[sel])) if sel.any() else np.nan))
    df = pl.DataFrame(rows)
    df.write_csv(a.out)
    wide = df.pivot(on="bin", index="method", values="crps_rel").sort("all")
    with pl.Config(tbl_rows=200, tbl_cols=10, float_precision=3, tbl_width_chars=200):
        print(wide)


if __name__ == "__main__":
    main()
