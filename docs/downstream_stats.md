# Downstream statistics: how every reported number is produced

Four numbers decide whether an integration run is good: merging statistics, refinement R-values, anomalous peak heights, and -- for the electric-field data only -- CCsym.
Both modes end at the same four, but the merging step differs because the two data types are scaled by different programs.

This file is the contract.
If a number appears in `results_by_dataset.md`, it came from one of the chains below, and anything produced another way is not comparable to it.

## The two chains

| | monochromatic | polychromatic (Laue) |
|---|---|---|
| scale + merge | `dials.scale` + `dials.merge` | `careless` |
| merging statistics | DIALS `merged.html` | careless's own tools |
| refinement | `phenix.refine` | `phenix.refine` |
| anomalous peaks | `rs.find_peaks` | `rs.find_peaks` **and** `anomalous_peak_heights.py` (see below) |
| symmetry response | -- | `rs.ccsym` (PDZ2 only) |

Refinement and peak finding are deliberately identical across modes, so a mono and a Laue run differ only where the physics forces them to.

## The reference files

Every path below was checked to exist on 2026-10-08.
A run that cannot find one of these does not fail loudly -- it produces numbers against whatever it does find -- so they are listed explicitly rather than left to the configs.

### Monochromatic (`scripts/sbgrid/`)

| role | where it comes from |
|---|---|
| starting model | downloaded from RCSB by the dataset card's PDB ID (`https://files.rcsb.org/download/{pdb}.pdb`), or `--model` |
| experiment list | `$DATA_DIR/dials_reference/<sweep>/integrated.expt` |
| free-flag donor | `$DATA_DIR/dials_reference/merged.mtz`, via `--rfree-from` |
| phenix | `/n/hekstra_lab_tier0/Lab/garden/phenix_1_20/phenix-1.20.1-4487/phenix_env.sh` |
| anomalous elements | `MN FE ZN CU NI CO SE S P I BR` (`refine.py:ANOMALOUS_ELEMENTS`) |

Per dataset the model is the deposited entry: 821 -> 7LVC, 845 -> 7RIN, 6VYN -> 6VYN, 140 -> 4U4H.

### Polychromatic (`scripts/poly/poly_pipeline_cfg.yaml`)

HEWL 1118 refines against the **monochromatic 9b7c model**; that is deliberate and correct, not a leftover.

| role | path |
|---|---|
| phenix starting model | `.../integrato_refac/integrator/reference_data/shaken_9b7c.pdb` |
| phenix pass 1 / pass 2 | `.../laue_analysis/custom_refinement_param_{1,2}_.eff` |
| peak-height model | `.../integrato_refac/integrator/reference_data/9b7c.pdb` |
| peak-height script | `/n/lab_storage/hekstra_lab/people/aldama/refltorch/scripts/laue_output/anomalous_peak_heights.py` |
| elements sampled | `[S,I]` |
| phenix | same `phenix_1_20` as the mono arm |

(`.../` is `/n/holylabs/hekstra_lab/Users/laldama`.)

## Monochromatic

Driver: `scripts/sbgrid/cannon/05_integrator_pipeline.sh <run-dir> [epoch]`.

```
predictions (pred.parquet)
  -> write_predictions.py        one .refl per sweep
  -> dials.scale  anomalous=True
  -> dials.merge  anomalous=True -> merged.mtz + merged.html
  -> scripts/mono/emit_merging_stats.py merged.html -> merging_stats.csv
  -> scripts/sbgrid/refine.py --rfree-from <reference merged.mtz>
       -> phenix.refine -> refined mtz/pdb, R-work and R-free
  -> rs.find_peaks <refined.mtz> <refined.pdb> -f ANOM -p PANOM -z 5.0 -o peaks.csv
```

Two details carry the comparison and are easy to lose.

**The experiment list comes from the reference**, not from our own indexing: each per-sweep `.refl` is scaled against `dials_reference/<sweep>/integrated.expt`.
That keeps the crystal models identical between arms, so the only difference is integration.

**The free set is copied, never redrawn.**
`dials.merge` writes its own `FreeR_flag` per run, so two arms merged separately have incomparable R-free.
`refine.py --rfree-from <donor>` copies the donor's flags and overrides any already present.

## Polychromatic (Laue)

Driver: `scripts/poly/run_pipeline.py --run-dir <run> --steps predict,careless,phenix,peaks`.

```
predictions -> preds_epoch_XXXX.mtz
  -> careless poly <config N flags> "BATCH,xcal,ycal,dHKL,wavelength"
       -> scaling/configN_0.mtz, configN_xval_0.mtz
  -> scripts/poly/emit_merging_stats.py --scaling-dir ... --config N --unmerged ...
       -> merging_stats.csv
  -> phenix.refine, two passes (pass 2 reuses pass 1's data and free flags)
       pass 1 is a WARM-UP against the shaken model and lands near R-free 0.34;
       pass 2 is the result. `refined_2.log` is the file to read.
  -> rs.find_peaks ... -f ANOM -p PHANOM -z 5.0   -> peaks.csv
  -> anomalous_peak_heights.py --elements "[S,I]"  -> peak_heights.csv
```

`careless` recipes live in `scripts/poly/careless_configs.py` and reproduce the original `config{1..6}.sh`; config 4 (anomalous + positional encoding) is the default.

Merging statistics come from careless's own tools rather than being recomputed, because they know how the half-datasets were split:

| quantity | tool |
|---|---|
| CC-half | `careless.cchalf` |
| CC-anom | `careless.ccanom` |
| I/sigma | `careless.isigi` |
| R-split | `careless.rsplit` |
| CCsym | `rs.ccsym` (PDZ2) |

R-pim is the exception: it needs multiplicity, so `emit_merging_stats.py` computes it from the unmerged input, the same quantity DIALS reports.

Note the phase column differs by mode: phenix writes `ANOM`/`PHANOM` here, and passing `-p PANOM` raises `KeyError`.

## The shared contract

Both `emit_merging_stats.py` scripts -- `scripts/mono/` and `scripts/poly/` -- write the **same** wide CSV, one row per resolution shell:

```
bin, d_max, d_min, n_obs, n_unique, cc_half, cc_anom, r_pim, i_over_sigma
```

That is what lets one figure place a DIALS-scaled monochromatic run and a careless-scaled Laue run on the same axes.
Anything that wants merging statistics should read `merging_stats.csv` and never parse `merged.html` or careless output directly.

## Per-epoch metrics

The figure code already supports trajectories; what has been missing is output in the layout it reads.

`plot_peaks.py` extracts the epoch from the path with the pattern `/epoch_(\d+)/`, so **every chain above must be run with its output under `<predictions_dir>/epoch_<NNNN>/`**.
Given that, it builds per-epoch peak heights and per-epoch R-work/R-free on its own: it parses `Final R-work = ..., R-free = ...` from the phenix log, and an epoch whose refinement failed leaves a gap in the curve rather than failing the figure.
`plot_merging.py` does the same for merging statistics, and references with no epoch structure are drawn as a flat line across the range.

Figures are then produced by `scripts/make_figures.sh --plot-cfg <cfg> --out-dir <dir>`, which runs loss, compare, merging and peaks from one manifest of runs.

Two requirements that are easy to miss when setting a run up:

**`save_top_k: -1`** in the training config, or there are no intermediate checkpoints to build a trajectory from.

**Both peak scripts run, and they answer different questions.**
`peaks:method: both` produces two files, and reading the wrong one silently gives the wrong answer.

| file | written by | what it reports |
|---|---|---|
| `peaks.csv` | `rs.find_peaks` | peaks at model atoms carrying density, in the schema `plot_peaks.py` parses (`seqid`, `peakz`) |
| `peak_heights.csv` | `anomalous_peak_heights.py` | the ANOM map sampled at the named sites, including the **iodides** |

On HEWL 1118 `rs.find_peaks` returns only cysteines and methionines; the headline IOD 204/205/206 values come from `peak_heights.csv`.
Comparing an arm's `peaks.csv` against a documented IOD number is comparing two different measurements.
`peak_heights.csv` is a transposed two-row CSV that `plot_peaks.py` cannot read, which is why per-epoch peak trajectories use `peaks.csv` while headline anomalous numbers use `peak_heights.csv`.
