"""Write the PRIOR-FREE maximum-likelihood intensities into the per-sweep MTZs.

careless models observations with a symmetric, unbounded likelihood, i.e. it expects
an unbiased measurement. The posterior mean is not one: measured against the
prior-free profile-fit estimate, the shrinkage slope is 0.081 for the weakest eighth
of reflections (where 70% of prior-free estimates are negative) and ~1.0 only above
I/sigma ~ 4, and the outer-shell weak tail is dragged from -0.167 Sigma to +0.233 Sigma.

So this arm hands careless the estimate it actually assumes: the same
variance-weighted profile fit laue-dials uses,

    w_p = (pi_p / v_p) / sum_p (pi_p^2 / v_p),   I = sum_p w_p (c_p - b),

built from OUR learned profile and background but WITHOUT the Wilson prior and
without laue-dials' positivity clamp. 8.71% of these are negative (14% in the outer
shell), which is the point: negative intensities are information, and they cancel
against positives when the merger averages, whereas a truncated estimator's bias
survives averaging.

Everything else matches the reference arm, including BATCH from poly_refined.expt.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from integrator.io import write_mtz_from_preds

M = Path("/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model")
POOL = M / "pooled"
REF = Path("/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref/gain_0,3_from_stills")

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default=str(M / "figs" / "shrinkage_audit.npz"))
ap.add_argument("--tree", required=True)
ap.add_argument("--floor", type=float, default=None,
                help="clamp I at this value, to isolate the effect of allowing negatives")
args = ap.parse_args()

tree = Path(args.tree)
a = np.load(args.npz)
I_ml = a["I_ml"].astype(np.float64)
S_ml = a["S_ml"].astype(np.float64)
bg = None

meta = torch.load(POOL / "metadata.pt", weights_only=False)
n_pool = len(meta["d"])
if len(I_ml) != n_pool:
    raise SystemExit(f"npz has {len(I_ml)} rows, pooled set has {n_pool}")
if not np.isfinite(I_ml).all() or not np.isfinite(S_ml).all():
    raise SystemExit("non-finite ML estimates")
if (S_ml <= 0).any():
    raise SystemExit(f"{(S_ml <= 0).sum()} non-positive sigmas; careless needs sigma > 0")

if args.floor is not None:
    n = int((I_ml < args.floor).sum())
    I_ml = np.maximum(I_ml, args.floor)
    print(f"clamped {n:,} reflections at I = {args.floor}")
print(f"I: frac<0 {(I_ml < 0).mean()*100:.2f}%  min {I_ml.min():.2f}  "
      f"median {np.median(I_ml):.2f}   sigma: median {np.median(S_ml):.2f}")

# qi_var is converted to SIGI by a square root inside the writer
ranges = yaml.safe_load((POOL / "sweep_ranges.yaml").read_text())["ranges"]
gdir = tree / "gain_0,3_from_stills"
total = 0
for s, (lo, hi) in ranges.items():
    d = gdir / f"dials_files_{s}"
    d.mkdir(parents=True, exist_ok=True)
    expt = REF / f"dials_files_{s}" / "poly_refined.expt"
    if not expt.exists():
        raise SystemExit(f"missing {expt}")
    write_mtz_from_preds(
        {
            "refl_ids": np.arange(lo, hi),
            "qi_mean": I_ml[lo:hi],
            "qi_var": S_ml[lo:hi] ** 2,
            "qbg_mean": np.zeros(hi - lo),
        },
        POOL / "metadata.pt",
        POOL,
        d / "integrated.mtz",
        expt_file=expt,
    )
    shutil.copy(REF / f"dials_files_{s}" / "rmsds.log", d / "rmsds.log")
    total += hi - lo
    print(f"  {s:10s} {hi - lo:>7,} refl -> {d / 'integrated.mtz'}")
print(f"total {total:,}")
print("SPLIT_ML_DONE")
