"""Whether the shoeboxes' pixels are what the likelihood assumes: independent Poisson about a flat background.

On shoeboxes whose true intensity is below `--max-itrue` photons (background
only, by the simulation's truth table, joined on (image_num, h, k, l)), for the
valid pixels c_k of each box:

    flat:    dispersion D = var(c) / mean(c)                         1 for Poisson about a constant
    plane:   residual r = c - (a + b x + c y), least squares on the box;
             D_plane = sum r^2 / (n - 3) / mean(c)                    1 for Poisson about a plane
             R2_plane = 1 - sum r^2 / sum (c - mean c)^2               variance a plane explains
    lag:     rho_x, rho_y = correlation of r with its right / lower neighbour   0 for independent pixels

reported as medians and quartiles over boxes, and binned by the box mean. A
flat-background Poisson likelihood is right when D, D_plane are about 1 and the
lags about 0; D > D_plane >= 1 with R2_plane > 0 means the background varies
within the box.

usage: python pixel_noise_check.py --data SHOEBOX_DIR --truth truth_reflections.csv [--max-itrue 1] [--n 20000]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prior_check import truth_for  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--max-itrue", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--side", type=int, default=25)
    a = ap.parse_args()

    from integrator.io import load_data
    meta = {k: np.asarray(v) for k, v in load_data(a.data / "metadata.npy").items()}
    I = truth_for(meta, a.truth)
    rows = np.flatnonzero(np.isfinite(I) & (I < a.max_itrue))
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(rows, size=min(a.n, len(rows)), replace=False))
    counts = np.load(a.data / "counts.npy", mmap_mode="r")
    masks = np.load(a.data / "masks.npy", mmap_mode="r")
    S = a.side
    yy, xx = np.mgrid[0:S, 0:S]
    X = np.stack([np.ones(S * S), xx.ravel() - S / 2, yy.ravel() - S / 2], 1)
    out = []
    for r in rows:
        c = np.asarray(counts[r], dtype=np.float64)
        m = np.asarray(masks[r]).astype(bool).reshape(-1)
        if m.sum() < 100:
            continue
        cv, Xv = c[m], X[m]
        mu = cv.mean()
        if mu <= 0:
            continue
        beta, *_ = np.linalg.lstsq(Xv, cv, rcond=None)
        res = cv - Xv @ beta
        D = cv.var(ddof=1) / mu
        Dp = (res**2).sum() / (len(cv) - 3) / mu
        R2 = 1 - (res**2).sum() / max(((cv - mu) ** 2).sum(), 1e-12)
        full = np.full(S * S, np.nan)
        full[m] = res
        g = full.reshape(S, S)

        def lag(u, v):
            ok = np.isfinite(u) & np.isfinite(v)
            return np.corrcoef(u[ok], v[ok])[0, 1] if ok.sum() > 50 else np.nan

        out.append(dict(mean=mu, D=D, D_plane=Dp, R2_plane=R2,
                        rho_x=lag(g[:, :-1].ravel(), g[:, 1:].ravel()), rho_y=lag(g[:-1].ravel(), g[1:].ravel())))
    df = pl.DataFrame(out)
    print(f"{df.height} background-only boxes (I_true < {a.max_itrue})")
    q = df.select([pl.col(c).quantile(x).alias(f"{c}_{int(100*x)}") for c in ("D", "D_plane", "R2_plane", "rho_x", "rho_y") for x in (0.25, 0.5, 0.75)])
    with pl.Config(tbl_cols=20, float_precision=3):
        print(q)
        print(df.with_columns(bin=pl.col("mean").qcut(5)).group_by("bin").agg(
            pl.len(), pl.col("mean").median(), pl.col("D").median(), pl.col("D_plane").median(),
            pl.col("R2_plane").median(), pl.col("rho_x").median(), pl.col("rho_y").median()).sort("mean"))


if __name__ == "__main__":
    main()
