"""PIT values of the true latents of an `sbc_simulate.py` dataset under a trained run's posteriors.

For each reflection j and latent x in {I, b}, with q(x | c_j) the Gamma the
run exports (shape m^2 / v, rate m / v from `q*_mean`, `q*_var`, exact for the
svae family's conjugate q):

    u_j = F_q(x_true,j)

If the posterior is calibrated, u is Uniform(0, 1) in every subset chosen by
quantities the inference conditions on (the counts, the metadata) or by the
truth. Departures, summarised per bin of I_true and of d:

    mean(u) - 1/2         > 0: the posterior sits below the truth
    12 var(u)             1 if calibrated; > 1 the posterior is too narrow, < 1 too wide
    sqrt(n) max|ECDF - u| the Kolmogorov statistic; above 1.36 rejects uniformity at 5 %
                          for one bin taken alone

`i_rb`, `var_rb` are a point estimate and a variance, not a posterior; for them
the same table is computed from u = Phi((I_true - i_rb) / sqrt(var_rb)).

Reflections are not independent replicates in the strict SBC sense: they share
the global parameters the run learned (the Wilson G and B, the profile prior).

usage: python sbc_ranks.py --truth DIR/sbc_truth.npz --pred 'GLOB/*.parquet'
           --method NAME --out DIR [--metadata DIR/metadata.npy]
Writes DIR/sbc_summary.csv (one row per method, estimator, latent, binning, bin)
and DIR/sbc_pit_hist.csv (20-bin PIT histograms on the same keys).
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import special, stats

I_EDGES = [0, 10, 30, 100, 300, 1000, 3000, math.inf]
D_EDGES = [1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0, 4.0, math.inf]


def gamma_pit(x, mean, var):
    shape = mean * mean / var
    return special.gammainc(shape, x * mean / var)


def summarise(u: np.ndarray) -> dict:
    u = np.sort(u[np.isfinite(u)])
    n = len(u)
    if n == 0:
        return dict(n=0, mean_minus_half=np.nan, var12=np.nan, ks=np.nan)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    D = max(np.max(ecdf_hi - u), np.max(u - ecdf_lo))
    return dict(n=n, mean_minus_half=float(u.mean() - 0.5), var12=float(12 * u.var()), ks=float(math.sqrt(n) * D))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--pred", required=True, help="glob of the prediction parquet files")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    t = np.load(a.truth)
    truth = pl.DataFrame({"refl_ids": t["refl_ids"].astype(np.int64), "I_true": t["I"], "b_true": t["b"]})
    files = sorted(glob.glob(a.pred))
    if not files:
        raise SystemExit(f"no parquet matches {a.pred}")
    pred = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    pred = pred.with_columns(pl.col("refl_ids").cast(pl.Int64))
    j = pred.join(truth, on="refl_ids", how="inner")
    if j.height != truth.height:
        print(f"warning: {j.height} of {truth.height} reflections matched")

    I, b = j["I_true"].to_numpy(), j["b_true"].to_numpy()
    pits = {("posterior", "I"): gamma_pit(I, j["qi_mean"].to_numpy(), j["qi_var"].to_numpy()),
            ("posterior", "b"): gamma_pit(b, j["qbg_mean"].to_numpy(), j["qbg_var"].to_numpy())}
    if "i_rb" in j.columns:
        z = (I - j["i_rb"].to_numpy()) / np.sqrt(j["var_rb"].to_numpy())
        pits[("rb", "I")] = stats.norm.cdf(z)

    d = j["d"].to_numpy()
    rows, hist = [], []
    for (est, lat), u in pits.items():
        for binning, x, edges in (("all", np.zeros_like(I), [-1, 1]), ("i_true", I, I_EDGES), ("d", d, D_EDGES)):
            idx = np.digitize(x, edges) - 1
            for k in range(len(edges) - 1):
                s = idx == k
                key = dict(method=a.method, estimator=est, latent=lat, binning=binning,
                           lo=float(edges[k]), hi=float(edges[k + 1]))
                rows.append(key | summarise(u[s]))
                h, _ = np.histogram(u[s], bins=20, range=(0, 1))
                hist += [key | dict(u_lo=i / 20, count=int(c)) for i, c in enumerate(h)]
    out = pl.DataFrame(rows)
    out.write_csv(a.out / "sbc_summary.csv")
    pl.DataFrame(hist).write_csv(a.out / "sbc_pit_hist.csv")
    with pl.Config(tbl_rows=200, tbl_cols=12):
        print(out.filter(pl.col("binning") != "d").select("estimator", "latent", "binning", "lo", "hi", "n",
                                                          "mean_minus_half", "var12", "ks"))


if __name__ == "__main__":
    main()
