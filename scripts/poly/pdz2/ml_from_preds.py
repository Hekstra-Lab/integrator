"""Prior-free profile-fit intensities from any arm's predicted profiles.

geom_both (+0.0214, 6.1 sigma) and prior-free ML intensities (+0.0237, 7.2 sigma) each
improve CCsym by the same amount and are statistically indistinguishable from each other.
Either they relieve the same bottleneck, in which case combining them changes nothing, or
they are independent and the effects add. This builds the combination: the same
variance-weighted profile fit laue-dials uses,

    w_p = (pi_p / v_p) / sum_p (pi_p^2 / v_p),   I = sum_p w_p (c_p - b),

but with geom_both's geometry-steered profiles rather than the baseline's round ones, and
no Wilson prior and no positivity clamp.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True, help="prediction dir holding a profile column")
ap.add_argument("--pool", default="/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/pooled")
ap.add_argument("--out", required=True, help="npz to write (I_ml, S_ml)")
a = ap.parse_args()

POOL = Path(a.pool)
cands = sorted(Path(a.pred).glob("**/pred.parquet")) + sorted(Path(a.pred).glob("**/preds_epoch_*.parquet"))
if not cands:
    raise SystemExit(f"no prediction parquet under {a.pred}")
pq = cands[-1]
lazy = pl.scan_parquet(pq)
cols = lazy.collect_schema().names()
if "profile" not in cols:
    raise SystemExit(f"{pq} has no profile column")
small = lazy.select(["qi_mean", "qbg_mean", "d"]).collect()

meta = torch.load(POOL / "metadata.pt", weights_only=False)
n = len(meta["d"])
if small.height != n:
    raise SystemExit(f"{pq}: {small.height} rows vs pooled {n}")
if not np.allclose(small["d"].to_numpy().astype(float),
                   np.asarray(meta["d"]).astype(float), atol=1e-4):
    raise SystemExit("prediction is not in pooled row order")

m = small["qi_mean"].to_numpy().astype(np.float64)
bg = small["qbg_mean"].to_numpy().astype(np.float64)
counts = np.load(POOL / "counts.npy", mmap_mode="r")
masks = np.load(POOL / "masks.npy", mmap_mode="r")
npix = counts.shape[1]

I_ml = np.empty(n)
S_ml = np.empty(n)
CH = 20_000
for lo in range(0, n, CH):
    hi = min(lo + CH, n)
    blk = lazy.slice(lo, hi - lo).select("profile").collect()["profile"].to_numpy()
    p = np.asarray(blk.tolist() if blk.dtype == object else blk, dtype=np.float64)
    p = p.reshape(len(p), npix)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    c = counts[lo:hi].astype(np.float64)
    msk = masks[lo:hi]
    bb = bg[lo:hi, None]
    rate = np.maximum(m[lo:hi, None] * p + bb, 1e-6)
    w = p / rate
    denom = np.maximum((p * w).sum(axis=1, keepdims=True), 1e-12)
    w = np.where(msk, w / denom, 0.0)
    I_ml[lo:hi] = (w * (c - bb)).sum(axis=1)
    S_ml[lo:hi] = np.sqrt(np.maximum((rate * w * w).sum(axis=1), 1e-12))

print(f"{pq.parent.name}/{pq.name}")
print(f"  I_ml: frac<0 {(I_ml < 0).mean()*100:.2f}%  min {I_ml.min():.2f}  "
      f"median {np.median(I_ml):.2f}")
print(f"  S_ml: median {np.median(S_ml):.2f}  nonpositive {(S_ml <= 0).sum()}")
np.savez(a.out, I_ml=I_ml, S_ml=S_ml, m=m, d=small["d"].to_numpy())
print(f"wrote {a.out}")
print("ML_FROM_PREDS_DONE")
