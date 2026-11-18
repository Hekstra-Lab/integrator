"""CCsym's own quantity, measured on UNMERGED intensities with no careless involved.

Merged CCsym has a run-level noise floor of ~0.016: careless fed identical input twice
differs by that much, so no effect below ~0.02 is interpretable from single runs, and
slicing into 90 bin/repeat cells does not help because the noise is shared across them.
This computes the same correlation from the per-observation intensities directly, where
there are ~10^5 pairs and no scaling model to fluctuate.

Definition, following pdz2_ref/scripts/custom_ccsym.py: split into half-datasets, in each
half form the internal difference between a reflection and its 2-fold mate, and correlate
the two halves' differences per resolution bin. In the raw C2 setting the 2-fold is simply
(h,k,l) -> (-h,k,-l), so no change of basis is needed; Friedel mates are folded first,
which is what the reference's hkl_to_asu() does after applying the operator.

The absolute value is lower than merged CCsym because there is no scale model here --
observations are only crudely normalized by a per-image median of the DIALS sum intensity,
a column every arm shares so it cannot favour one intensity definition. What is comparable
is the arm-to-arm DIFFERENCE, since the arms differ in nothing but the intensity column,
and the scatter over independent half-splits gives its error bar directly.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

M = Path("/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model")
POOL = M / "pooled"
SWEEPS = [f"{p}_{t}" for p in "cdef" for t in ("off", "50ns", "100ns", "200ns")]

ap = argparse.ArgumentParser()
ap.add_argument("--delay", default="200ns", choices=["off", "50ns", "100ns", "200ns"])
ap.add_argument("--bins", type=int, default=10)
ap.add_argument("--min-obs", type=int, default=2)
ap.add_argument("--reps", type=int, default=12)
a = ap.parse_args()

meta = torch.load(POOL / "metadata.pt", weights_only=False)
H = np.asarray(meta["H"]).astype(np.int64)
K = np.asarray(meta["K"]).astype(np.int64)
L = np.asarray(meta["L"]).astype(np.int64)
d = np.asarray(meta["d"]).astype(np.float64)
sweep = np.asarray(meta["sweep_id"]).astype(np.int64)
img = np.asarray(meta["image_num"]).astype(np.int64)
i_sum = np.asarray(meta["intensity.sum.value"]).astype(np.float64)

aud = np.load(M / "figs" / "shrinkage_audit.npz")
ARMS = {"posterior": aud["m"].astype(np.float64), "ML_prior_free": aud["I_ml"].astype(np.float64)}
if len(aud["m"]) != len(H):
    raise SystemExit("audit rows do not match the pooled set")

keep = np.isin(sweep, [i for i, s in enumerate(SWEEPS) if s.endswith(a.delay)])
print(f"delay {a.delay}: {keep.sum():,} observations")

# per-image scale from a column common to every arm
img_key = sweep * 1000 + img
uk, inv = np.unique(img_key, return_inverse=True)
med = np.ones(len(uk))
o = np.argsort(inv, kind="stable")
b = np.searchsorted(inv[o], np.arange(len(uk)))
b = np.append(b, len(o))
for j in range(len(uk)):
    v = i_sum[o[b[j]:b[j + 1]]]
    if len(v):
        mm = np.median(v)
        med[j] = mm if mm > 0 else 1.0
scale = med[inv]

# fold Friedel, then pair through the 2-fold (-h, k, -l)
def fold(x):
    neg = -x
    flip = (x[:, 2] < 0) | ((x[:, 2] == 0) & ((x[:, 1] < 0) | ((x[:, 1] == 0) & (x[:, 0] < 0))))
    return np.where(flip[:, None], neg, x)


def pack(x):
    return (x[:, 0] + 512) * 1_048_576 + (x[:, 1] + 512) * 1024 + (x[:, 2] + 512)


hkl = np.stack([H, K, L], axis=1)
canon = fold(hkl)
mate = fold(np.stack([-canon[:, 0], canon[:, 1], -canon[:, 2]], axis=1))
key, mkey = pack(canon), pack(mate)

idx = np.where(keep)[0]
gid_all, gidx = np.unique(key[idx], return_inverse=True)
n_g = len(gid_all)
print(f"unique P1 reflections: {n_g:,}")

# group id of each group's 2-fold mate
rep_row = np.zeros(n_g, dtype=np.int64)
rep_row[gidx] = np.arange(len(idx))
mate_key_per_group = mkey[idx[rep_row]]
pos = np.searchsorted(gid_all, mate_key_per_group)
pos = np.clip(pos, 0, n_g - 1)
has_mate = gid_all[pos] == mate_key_per_group
mate_gid = np.where(has_mate, pos, -1)
# keep each pair once
pair_lo = np.where(has_mate & (np.arange(n_g) < mate_gid), np.arange(n_g), -1)
pairs = pair_lo[pair_lo >= 0]
print(f"2-fold pairs present on both sides: {len(pairs):,}")

dres = np.zeros(n_g)
np.add.at(dres, gidx, d[idx])
cnt_all = np.bincount(gidx, minlength=n_g)
dres /= np.maximum(cnt_all, 1)

s2p = 1.0 / (0.5 * (dres[pairs] + dres[mate_gid[pairs]])) ** 2
edges = np.quantile(s2p, np.linspace(0, 1, a.bins + 1))
rng = np.random.default_rng(0)

results = {}
for name, I in ARMS.items():
    Iw = (I / scale)[idx]
    per_bin = np.full((a.bins, a.reps), np.nan)
    for rep in range(a.reps):
        half = rng.integers(0, 2, len(idx))
        m1 = half == 0
        s1 = np.bincount(gidx[m1], weights=Iw[m1], minlength=n_g)
        c1 = np.bincount(gidx[m1], minlength=n_g)
        s2_ = np.bincount(gidx[~m1], weights=Iw[~m1], minlength=n_g)
        c2 = np.bincount(gidx[~m1], minlength=n_g)
        ok_g = (c1 >= a.min_obs) & (c2 >= a.min_obs)
        mean1 = np.where(ok_g, s1 / np.maximum(c1, 1), np.nan)
        mean2 = np.where(ok_g, s2_ / np.maximum(c2, 1), np.nan)
        pa, pb = pairs, mate_gid[pairs]
        good = ok_g[pa] & ok_g[pb]
        dA = mean1[pa] - mean1[pb]
        dB = mean2[pa] - mean2[pb]
        if rep == 0:
            print(f"{name:14s} usable pairs {int(good.sum()):,}")
        for i in range(a.bins):
            m = good & ((s2p >= edges[i]) & (s2p < edges[i + 1]) if i < a.bins - 1
                        else (s2p >= edges[i]))
            if m.sum() > 50:
                per_bin[i, rep] = spearmanr(dA[m], dB[m]).statistic
    results[name] = per_bin

names = list(results)
print(f"\n{'d range (A)':>16}" + "".join(f"{n:>22s}" for n in names))
for i in range(a.bins):
    lo = 1 / np.sqrt(edges[i + 1])
    hi = 1 / np.sqrt(edges[i])
    row = f"{hi:7.2f}-{lo:6.2f} "
    for n in names:
        row += f"{np.nanmean(results[n][i]):+14.4f}+/-{np.nanstd(results[n][i]):6.4f}"
    print(row)

print(f"\n{'MEAN over bins':>16}" + "".join(
    f"{np.nanmean(results[n]):+14.4f}+/-{np.nanstd(np.nanmean(results[n], axis=0)):6.4f}"
    for n in names))
if len(names) == 2:
    per_rep_diff = np.nanmean(results[names[1]], axis=0) - np.nanmean(results[names[0]], axis=0)
    print(f"\n{names[1]} - {names[0]} = {per_rep_diff.mean():+.4f} "
          f"+/- {per_rep_diff.std(ddof=1):.4f} (sd over {a.reps} half-splits)")
    print(f"  the half-split scatter is the ONLY noise here; there is no careless run")
print("UNMERGED_CCSYM_DONE")
