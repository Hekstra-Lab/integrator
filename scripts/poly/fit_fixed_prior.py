"""Fit a FIXED intensity prior once, before training, from per-reflection estimates that do not use a prior (two-stage empirical Bayes).

Stage 1 is any export with a prior-free intensity estimate per reflection; the
default column is `i_rb`, the profile-weighted least-squares fit with a
Schur-complement variance, which simulation-based calibration found centred on
the truth when the global parameters are right. Its mean in a cell of
(wavelength, resolution) estimates the prior mean there, since E[i_rb] = E[I]
for an unbiased estimator, whatever its noise.

Stage 2 fits, on cells of `--lambda-bins` equal wavelength bins over
[lambda_min, lambda_max] and the dataset's resolution bins (`group_labels_<n>.npy`,
the bins the data module injects as `group_label`), with cell mean m_ak and count
n_ak (cells with n < `--min-count` or m <= 0 are dropped):

    Wilson:   log m_ak = sum_k c_k T_k(x(lambda_a)) - 2 B s2_ak      weighted least squares, weights n_ak
              (x = (lambda - lam_mid) / lam_scale, s2 = mean 1/(4 d^2) of the cell); written as
              wilson_prior.pt {c, lam_mid, lam_scale, raw_B} in the format `spectrum_init_from` reads
    separable: log m_ak = g_a + h_k                                   the alternating fit of
              `polychromatic_wilson_eb`'s M-step, h centred; written as eb_prior.pt {eb_g, eb_h}

Train with the prior held: `polychromatic_wilson` with `learn_G: false`,
`learn_B: false`, `spectrum_init_from: .../wilson_prior.pt`, or
`polychromatic_wilson_eb` with `eb_init_from: .../eb_prior.pt`, `eb_refit: false`.

usage: python fit_fixed_prior.py --pred PRED.parquet --data DATA_DIR --out-dir DIR
           [--column i_rb] [--degree 5] [--lambda-min 1.01] [--lambda-max 1.19]
           [--lambda-bins 36] [--n-bins 30] [--min-count 30]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch


def cells(lam, gl, lo, hi, n_lam):
    a = np.clip(((lam - lo) / (hi - lo) * n_lam).astype(np.int64), 0, n_lam - 1)
    return a, gl


def separable(S, N, n_iter=50):
    """g, h maximising sum n (log-mean model) on cell sums S and counts N; h centred by count."""
    ok = N > 0
    g = np.log(np.maximum(S.sum(1) / np.maximum(N.sum(1), 1), 1e-8))
    h = np.zeros(S.shape[1])
    for _ in range(n_iter):
        nh = N.sum(0)
        h = np.where(nh > 0, np.log(np.maximum((S * np.exp(-g)[:, None]).sum(0) / np.maximum(nh, 1), 1e-8)), h)
        ng = N.sum(1)
        g = np.where(ng > 0, np.log(np.maximum((S * np.exp(-h)[None, :]).sum(1) / np.maximum(ng, 1), 1e-8)), g)
    wk = N.sum(0)
    shift = (h * wk).sum() / max(wk.sum(), 1)
    del ok
    return g + shift, h - shift


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--column", default="i_rb")
    ap.add_argument("--degree", type=int, default=5)
    ap.add_argument("--lambda-min", type=float, default=1.01)
    ap.add_argument("--lambda-max", type=float, default=1.19)
    ap.add_argument("--lambda-bins", type=int, default=36)
    ap.add_argument("--n-bins", type=int, default=30)
    ap.add_argument("--min-count", type=int, default=30)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    from integrator.io import load_data
    meta = load_data(a.data / "metadata.npy")
    row_of = {int(r): i for i, r in enumerate(np.asarray(meta["refl_ids"]))}
    glab = np.asarray(load_data(a.data / f"group_labels_{a.n_bins}.npy")).astype(np.int64)
    p = pl.read_parquet(a.pred)
    rows = np.array([row_of[int(r)] for r in p["refl_ids"]])
    I = p[a.column].to_numpy().astype(np.float64)
    lam = p["wavelength"].to_numpy().astype(np.float64)
    s2 = 1 / (4 * p["d"].to_numpy().astype(np.float64) ** 2)
    ai, ki = cells(lam, glab[rows], a.lambda_min, a.lambda_max, a.lambda_bins)
    nl, nd = a.lambda_bins, a.n_bins
    S, N, S2 = np.zeros((nl, nd)), np.zeros((nl, nd)), np.zeros((nl, nd))
    np.add.at(S, (ai, ki), I)
    np.add.at(N, (ai, ki), 1)
    np.add.at(S2, (ai, ki), s2)
    ok = (N >= a.min_count) & (S > 0)
    Sw, Nw = np.where(ok, S, 0), np.where(ok, N, 0)
    print(f"{int(ok.sum())} of {nl * nd} cells used, {int(Nw.sum())} of {len(I)} reflections")

    # separable (EB) form
    g, h = separable(Sw, Nw)
    torch.save({"eb_g": torch.tensor(g), "eb_h": torch.tensor(h), "column": a.column, "pred": str(a.pred)},
               a.out_dir / "eb_prior.pt")

    # Wilson form: log m = sum c_k T_k(x) - 2 B s2
    lam_mid, lam_scale = (a.lambda_min + a.lambda_max) / 2, (a.lambda_max - a.lambda_min) / 2
    lam_c = a.lambda_min + (np.arange(nl) + 0.5) * (a.lambda_max - a.lambda_min) / nl
    A_list, y, w = [], [], []
    for ia in range(nl):
        for ik in range(nd):
            if not ok[ia, ik]:
                continue
            x = np.clip((lam_c[ia] - lam_mid) / lam_scale, -1, 1)
            T = np.polynomial.chebyshev.chebvander(x, a.degree)[0]
            A_list.append(np.concatenate([T, [-2 * S2[ia, ik] / N[ia, ik]]]))
            y.append(math.log(S[ia, ik] / N[ia, ik]))
            w.append(N[ia, ik])
    A, y, w = np.array(A_list), np.array(y), np.sqrt(np.array(w))
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    c, B = coef[:-1], float(coef[-1])
    resid = y - A @ coef
    raw_B = math.log(math.expm1(max(B - 1.0, 1e-3)))  # B = softplus(raw_B) + b_min (b_min 1.0 in these configs)
    torch.save({"c": torch.tensor(c, dtype=torch.float32), "lam_mid": torch.tensor(lam_mid), "lam_scale": torch.tensor(lam_scale),
                "raw_B": torch.tensor(raw_B, dtype=torch.float32), "B": B, "column": a.column, "pred": str(a.pred)},
               a.out_dir / "wilson_prior.pt")
    wr = np.sqrt(np.average(resid**2, weights=w**2))
    print(f"Wilson fit: B = {B:.2f}, c = {np.round(c, 3).tolist()}, weighted rms log-residual {wr:.3f}")
    print(f"separable fit: h range {h.min():+.3f}..{h.max():+.3f}, g mean {g.mean():.3f}")
    print(f"wrote {a.out_dir / 'wilson_prior.pt'} and {a.out_dir / 'eb_prior.pt'} (b_min 1.0 assumed for raw_B)")


if __name__ == "__main__":
    main()
