# The PDZ2 / 1116 chain

Recovered from `laue_runs/pdz2_model/` on netscratch on 2026-11-18, where it
had been the only copy.
These produced every PDZ2 number the project reports, so the results were not
reproducible from a checkout until now.

The chain runs in this order. `scripts/poly/pdz2/arm_pdz2.sbatch <arm>` drives
steps 4 to 8 for one trained arm, and is the entry point to use.

| # | stage | script | tool it drives |
|---|---|---|---|
| 1 | reference integration, 16 sweeps | `../laue_process.sh` | laue-dials |
| 2 | shoeboxes | `refltorch.mksbox-laue` | refltorch |
| 3 | pool the 16 sweeps into one training set | `pool_pdz2.py` | -- |
| 4 | train | `../train.slurm` with `CONFIG=` | integrator |
| 5 | predict, last checkpoint | `arm_pdz2.sbatch` | integrator |
| 6 | split predictions into 16 per-sweep MTZs | `split_pdz2.py` | -- |
| 7 | frame cuts and concatenation | `expt_concat.py` **(deposited)** | laue-dials |
| 8 | scale and merge, 8 files jointly | `careless-cdef-ohp-mlpw.sh` **(deposited)** | careless |
| 9 | CCsym and CCpred | `custom_ccsym.py`, `stats_for_production.sh` **(deposited)** | careless, rs |
| 10 | weighted difference map | `wdm.py` **(deposited)**, then `wdm_peaks_compare.py` | rs-booster |
| 11 | refinement endpoint | `pdz2_refine_prep.py`, `refine_pdz2.sbatch`, `collect_rfree.py` | phenix |

**Steps 7 to 10 marked "deposited" are not ours.** They are the rs-station
laue-dials tutorial's own scripts, run verbatim so that integration is the
only thing that differs from the published result. They are copied into each
arm's tree from `laue_runs/pdz2_ref/scripts/` and are **not** vendored here;
see `docs/methods.md` for what that costs.

## The variant arms

Not every arm is a training run. Three rewrite a trained arm's intensities
and re-run steps 7 to 10 unchanged, so integration is still the only variable:

- `ml_from_preds.py` with `split_pdz2_ml.py` and `ml_arm.sbatch` -- the
  prior-free profile fit, the arm that beats `laue.integrate` on R-free.
  Superseded by `integrator.predict --rb-export`, which computes the same
  estimator in maintained code with a better sigma; this is kept because it
  produced the published number.
- `build_swap_trees.py` with `swap_arm.sbatch` -- crosses our intensities
  with our sigmas to decompose which of the two carries a difference.
- `wdm_arm.sbatch` -- the difference map alone, for an arm already merged.

## Tools

`unmerged_ccsym.py` measures CCsym's own quantity on unmerged intensities,
without careless in the path, as a control on the merged statistic.

`fix_manifest_stats.py` recomputes a dataset manifest's Anscombe statistics
from the counts. It found 1118 and 1117 wrong by 1.85x and 1.47x in sd and
PDZ2 in agreement with its own counts, which is why a predict pass on an old
PDZ2 checkpoint is safe and one on 1118 is not.
