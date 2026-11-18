"""Split pooled integrator predictions into the 16 per-sweep MTZs the reference
pipeline consumes.

The parquet's refl_ids are each sweep's OWN DIALS ids (they restart at 0 per
sweep), so rows are split positionally: the parquet is written in pooled dataset
order, which this asserts before splitting, and sweep_ranges.yaml gives each
sweep's row range. rmsds.log is copied from the reference tree because expt_concat
derives the pass-d bad-frame list from it.

`--estimator` picks which intensities are written. One training gives two
sets and which wins is not monotone across arms, so both are scored: `model`
is the integrator's posterior (`qi_mean`/`qi_var`) and `rb` is the
Rao-Blackwellized profile fit computed at export (`i_rb`/`var_rb`). The RB
columns exist only if the predict pass asked for them in `predict_keys` --
see scripts/poly/rb_predict_config.py -- because the fit happens at export
and not during training. The default is `model`, so a command written before
this option behaves exactly as it did.

BATCH comes from each sweep's `poly_refined.expt`, the same experiment list
`laue.integrate` was given, so it carries laue-dials' convention
(`BATCH = experiment index + 1`). Writing `image_num + 1` instead -- which is what
this did until the convention was checked -- silently disagrees wherever upstream
refinement dropped an experiment: on PDZ2 that was up to 98% of the reflections in
a sweep. Because `expt_concat.py` applies the tutorial's frame cuts BY BATCH and
careless takes BATCH as a per-image scaling key, the wrong labels both cut the
wrong frames and corrupt the scale model. See integrator.io.expt_batch.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

from integrator.io import write_mtz_from_preds

M = Path("/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model")
POOL = M / "pooled"
REF = Path("/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref/gain_0,3_from_stills")

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--tree", required=True)
ap.add_argument("--legacy-batch", action="store_true",
                help="write BATCH = image_num + 1, the pre-fix behaviour")
ap.add_argument("--estimator", choices=("model", "rb"), default="model",
                help="which intensities to write: the posterior (model) or "
                     "the profile fit computed at export (rb)")
args = ap.parse_args()

run, tree = Path(args.run_dir), Path(args.tree)
paths = yaml.safe_load((run / "run_paths.yaml").read_text())
pdir = Path(paths.get("predictions_dir") or (run / "predictions"))

cands = sorted(pdir.glob("**/pred.parquet")) + sorted(pdir.glob("**/preds_epoch_*.parquet"))
if not cands:
    raise SystemExit(f"no prediction parquet under {pdir}")
pq = cands[-1]
df = pl.read_parquet(pq)
print(f"predictions: {pq}  rows {df.height:,}  cols {df.columns}")

meta_pool = torch.load(POOL / "metadata.pt", weights_only=False)
if df.height != len(meta_pool["d"]):
    raise SystemExit(f"parquet has {df.height} rows, pooled set has {len(meta_pool['d'])}")
for k in ("d", "wavelength", "H", "K", "L"):
    if not np.allclose(df[k].to_numpy().astype(float),
                       np.asarray(meta_pool[k]).astype(float), atol=1e-4):
        raise SystemExit(f"parquet is not in pooled row order (mismatch in {k})")
print("verified: parquet is in pooled row order")

# `qbg_mean` is the background either way; only the intensity and its
# variance change with the estimator
IKEY, VKEY = (("i_rb", "var_rb") if args.estimator == "rb"
              else ("qi_mean", "qi_var"))
want = {IKEY, VKEY, "qbg_mean"}
cols = {k: df[k].to_numpy() for k in want if k in df.columns}
missing = want - set(cols)
if missing:
    raise SystemExit(
        f"parquet missing {sorted(missing)} for --estimator {args.estimator}; "
        f"has {df.columns}. The RB columns are produced only when the predict "
        f"pass lists them in predict_keys (scripts/poly/rb_predict_config.py)."
    )
print(f"estimator: {args.estimator}  ({IKEY}, {VKEY})")

ranges = yaml.safe_load((POOL / "sweep_ranges.yaml").read_text())["ranges"]
gdir = tree / "gain_0,3_from_stills"
print(f"BATCH source: {'image_num + 1 (LEGACY)' if args.legacy_batch else 'poly_refined.expt'}")
total = 0
for s, (lo, hi) in ranges.items():
    idx = np.arange(lo, hi)
    d = gdir / f"dials_files_{s}"
    d.mkdir(parents=True, exist_ok=True)
    expt = None if args.legacy_batch else REF / f"dials_files_{s}" / "poly_refined.expt"
    if expt is not None and not expt.exists():
        raise SystemExit(f"missing {expt}; BATCH cannot follow the laue-dials convention")
    write_mtz_from_preds(
        {"refl_ids": idx, **{k: v[lo:hi] for k, v in cols.items()}},
        POOL / "metadata.pt",
        POOL,
        d / "integrated.mtz",
        expt_file=expt,
        intensity_key=IKEY,
        variance_key=VKEY,
    )
    shutil.copy(REF / f"dials_files_{s}" / "rmsds.log", d / "rmsds.log")
    total += len(idx)
    print(f"  {s:10s} {len(idx):>7,} refl -> {d / 'integrated.mtz'}")
print(f"total {total:,} / {df.height:,}")
print("SPLIT_DONE")
