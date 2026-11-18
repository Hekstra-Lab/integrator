# What a run needs before every figure can be drawn

The reported figure list is in `README.md` beside this file.
This is the other half of it: for each figure, whether it can be drawn on a monochromatic run, on a polychromatic one, and what is missing where it cannot.

Two kinds of gap, kept apart because they cost different things:

- **a value that does not exist upstream** -- the reference integrator never computed it, or the export never wrote it.
  No script here can supply it; it needs a different export, a different tool, or a decision that the comparison is not available.
- **code that is not written** -- the measurement is on disk and nothing reduces it to the long-format CSV a figure reads.
  This is work in `scripts/figures/`, not on the cluster.

Checked against the cluster on 2026-11-17: `sim_runs/m9_*` for monochromatic, `ablate2/q*` (20 arms, hewl_1118) for polychromatic, and `laue_runs/1118_hewl3/integrated.refl` for the Laue reference.

## The figure set

| figure | monochromatic | polychromatic | what is missing |
|---|---|---|---|
| input shoeboxes against the predicted profiles | drawn | **blocked** | no poly parquet carries `qp_mean` |
| training loss, train against validation | drawn | data on disk | `tidy_poly.py` |
| training loss, test split | **blocked** | **blocked** | `loss_history.csv` has train and val only |
| model intensity against `intensity.prf.value` | drawn | **not available** | laue-dials computes no profile-fitted intensity |
| model intensity against `intensity.sum.value` | needs a redump | needs a dump | the column is in both parquets |
| model background against the reference | drawn | **partly available** | laue-dials writes `background.sum.value`, not `background.mean` |
| the same correlations within resolution shells | drawn | with the scatters | as above |
| merging statistics by resolution shell | drawn | data on disk | `tidy_poly.py` |
| `careless.stats`: CChalf, CCanom, CCpred, Rsplit, $I/\sigma_I$, completeness | not applicable | **drawn**: 1 of 20 on 1118, 6 of 6 arms on PDZ2 | run the `stats` step on the other 19 |
| the Wilson prior's scale and $B$ over training | drawn | data on disk | `tidy_poly.py`; `plots/wilson_params.csv` exists on every run |
| every component of the learned profile basis | drawn | drawn | -- |
| PDZ2: $CC_\mathrm{sym}$ by resolution shell | not applicable | drawn, 6 arms | -- |
| refinement $R_\mathrm{work}$ and $R_\mathrm{free}$ | drawn | data on disk | `tidy_poly.py` |
| anomalous peak height, one panel per site | drawn | data on disk | `tidy_poly.py`; atom names need the refined PDB |
| mean anomalous peak height | drawn | data on disk | `tidy_poly.py` |
| the learned spectrum $G(\lambda)$ | not applicable | drawn | -- |
| profile shape against the spots beside it | drawn | never measured | the geometry investigation ran on 9b7c only |
| arm comparison at one checkpoint | drawn | **blocked** | `tidy_xtal.sh` reads `dialsarm_*.log`, which the poly arm does not write |
| the six LaTeX tables | drawn | **not applicable as written** | they read `dials.scale` and `dials.merge`; the poly arm scales with careless |

"Data on disk" means the measurement exists in all 20 poly arms and only the reduction is missing.

## Closed on 2026-11-18

Three gaps turned out to be reductions that had never been written rather
than measurements that were never made, so they cost no cluster time beyond
a read.

- **The Wilson prior's scale and $B$-factor** have been in every run's
  `plots/wilson_params.csv` since `WilsonParamLogger` was added, on both
  arms, and were never pulled. `tidy_mono.py` now writes them.
- **The learned profile basis over training** is a parameter of
  `surrogates.qp.decoder`, so every checkpoint holds it.
  `ProfileBasisLogger` writes the same snapshots during training but is
  gated behind `integrator.train --figures` and no production run passed
  it, so `dump_basis.py` recovers the trajectory from the checkpoints
  instead. No retraining, no GPU, no data.
- **PDZ2 had no figures at all** and no `careless.stats` numbers. The
  `stats` step could not reach it because that chain scaled outside this
  pipeline, so `run_pipeline.py` gained `--out-base` and PDZ2 now has
  CChalf, Rsplit, CCpred, $I/\sigma_I$ and completeness on all six arms,
  plus CCsym by shell.

One thing that is still only recoverable going forward: `TrackedShoeboxLogger`
and `LatentSpaceLogger` sit behind the same `--figures` gate, and unlike the
basis their output cannot be reconstructed afterwards, because it depends on
which reflections were selected at the time.

## Values that are missing upstream

### The Laue reference has no profile-fitted intensity and no mean background

`laue_runs/1118_hewl3/integrated.refl` carries exactly these columns:

```
BATCH  H  K  L  PARTIAL
background.sum.value
intensity.sum.value  intensity.sum.variance
s1.0  s1.1  s1.2
xyzcal.px.0  xyzcal.px.1  xyzcal.px.2
```

So on the polychromatic arm:

- **`intensity.prf.value` does not exist.**
  There is no profile-fitted intensity in the Laue reference at all, so `intensity_prf` is a monochromatic quantity and the poly comparison is against the summation integrator only.
  This is not an export omission; `laue.integrate` does not produce one.
- **`background.mean` does not exist, but `background.sum.value` does.**
  This corrects the standing note that the polychromatic background comparison is blocked because `laue.integrate` reports no background.
  It reports a background, summed over the region it integrated rather than averaged per pixel.

  The model's `qbg_mean` is a per-pixel mean, so the two are not on one axis without the pixel count that the sum was taken over, and that count is not exported.
  `laue.integrate` v0.5.1 sums flat inside an ellipse whose area varies per reflection, so a fixed shoebox volume is the wrong divisor.
  What is available without any new export is the **correlation** between the model background and `background.sum.value`, which answers whether the model tracks the reference; what is not available is a scatter on equal axes, which would answer whether it agrees in magnitude.
  Getting the second needs the ellipse area, or the per-pixel background, out of `laue.integrate`.

### No prediction export on the poly arm carries the profile

Checked every parquet in `ablate2/q2_profile_geomboth`: `predictions/epoch_0000/test_preds_all.parquet` (14 columns), the nested `epoch_0019/pred.parquet` (14), and the rb export `predictions/rb/epoch_0019/pred.parquet` (19, which adds `n_eff`, `i_rb`, `var_rb`, `var_rb_cond`, `var_rb_profile`).
None has `qp_mean`.
The monochromatic parquet has it, along with every other column the figures want.

`plot_profile_grid.py --dump` reads the profile from the parquet rather than from a checkpoint, so the shoebox figure needs a predict pass with `qp_mean` in `predict_keys` and nothing more.

**That pass loads a checkpoint, and this is the risk to check first.**
The `ablate2` runs are from October, and the Anscombe manifest statistics were wrong for every `mksbox_laue` dataset and were fixed at source; an old checkpoint needs the old statistics.
A predict pass that normalizes its input differently than training did produces wrong profiles without failing, so the manifest must be compared against the run's own `config_log.yaml` before the export, not after.

### `loss_history.csv` has no test split

Both arms.
The test reflections are the exported ones, so a test loss needs an evaluation pass over them rather than a column in the training log.
The figure is written to draw three splits and draws two.

### `rs.find_peaks` writes an empty atom name

Both arms.
The fork in use has a `name` column and leaves it empty on every row, so the atom carrying each peak is resolved against the refined PDB instead.
That PDB is beside the peak list on the cluster and is not pulled by default: `PEAK_RUN=<cluster run directory> bash scripts/figures/pull_figdata.sh peak_models`, which has no default because the local mirror under `docs/figures/data/runs` does not record where it came from.
Without it the per-site panels title themselves by residue and number, which is what they are meant to say anyway; the atom only matters where a residue holds more than one candidate heavy atom.

## Code that is missing

### `tidy_poly.py`

The single largest gap, and it unblocks five figures at once.
`tidy_mono.py` hardcodes the monochromatic chain's paths -- `$S/dials9b7c/<arm>_<est>/merge.log`, `$S/refine9b7c/<arm>_<est>_<variant>/`, `$S/sim_runs/<arm>/plots/loss_history.csv` -- and the poly chain puts the same measurements elsewhere:

| what | where, per poly arm |
|---|---|
| training loss | `<arm>/plots/loss_history.csv`, the same columns as mono |
| merging by shell | `<arm>/predictions/epoch_0000/scaling/merging_stats.csv` |
| `careless.stats` | `<arm>/predictions/epoch_0000/scaling/stats/careless_stats.csv` |
| refinement | `<arm>/predictions/epoch_0000/scaling/config4_refine/refine1`, `refine2` |
| anomalous peaks | `.../config4_refine/peaks.csv` and `peak_heights.csv` |
| CCanom | `.../config4_refine/ccanom.csv` |
| the reference | `ablate/baseline_laue_dials`, the same layout |

Every arm's careless output is under `epoch_0000`, whatever epoch the checkpoint was.
`run_pipeline.py` resolves the last checkpoint and so looks under `epoch_0019` and finds nothing, which is why the `stats` step needs `--epoch 0` on these runs.

### Poly targets in `pull_figdata.sh`

There are none.
`mono`, `scatter`, `dials_logs` and `arm_metrics` are all monochromatic; `careless_stats` pulls one file from one job directory rather than per arm.

### The poly table set

The six tables in `docs/tables/` read `dials.scale` and `dials.merge`, which the poly arm never runs.
Its counterpart is `careless_stats.csv` plus `merging_stats.csv`, and no table is built from those yet.
`statistic_labels.csv` already carries the header, unit and format for the quantities the two arms share.

### Arm comparison on the poly arm

`tidy_xtal.sh` reads the per-arm DIALS logs.
The poly equivalent would read `careless_stats.csv` and the refinement logs, which is `tidy_poly.py` again.

## What one run must produce

For a monochromatic run, everything in the list is reachable today: the parquet carries `qp_mean`, both DIALS intensities and the background, and `tidy_mono.py` plus `pull_figdata.sh` cover the rest.
The one action outstanding is a redump of `scatter_points.csv` so that `intensity_sum` has data.

For a polychromatic run, in the order that unblocks the most per step:

1. **`predict_keys` must include `qp_mean`** at export, or the shoebox figure cannot be drawn for that run.
   This is the only item that must be decided before training output is thrown away, and on existing runs it means a predict pass behind the Anscombe-manifest check.
2. **Run the `stats` step** after careless, with `--epoch 0` while the export writes to `epoch_0000`.
   One arm is done; it reads MTZs that already exist and is CPU work.
3. **Write `tidy_poly.py`** and give `pull_figdata.sh` poly targets.
4. **Accept that two comparisons are not available**: `intensity_prf`, which the Laue reference does not compute, and the background on equal axes, which needs the ellipse area out of `laue.integrate`.
   The background correlation is available now.
