# Laue (polychromatic) end-to-end run

Train → predict → careless → phenix.refine (two passes) → anomalous peaks,
on `hewl_1118`.

## The storage migration

Lab storage moved: `/n/hekstra_lab/...` → `/n/lab_storage/hekstra_lab/...`.
Old paths baked into `dataset.yaml`, `processing_config.yaml`, and the configs
are what produce

```
OSError: Cannot open file for reading: ".../reflections_.refl"
```

Applied across this repo (173 lines, 58 files) with:

```bash
grep -rl '/n/hekstra_lab/' configs/ scripts/ | xargs sed -i '' 's|/n/hekstra_lab/|/n/lab_storage/hekstra_lab/|g'
```

`/n/holylabs` and `/n/hekstra_lab_tier0` are separate filesystems and did not
move; the `.eff` files, reference models, and the phenix 1.20 garden still live
there.

## Order of operations

| # | Step | Command |
| --- | --- | --- |
| 1 | Pre-flight | `python scripts/poly/preflight.py --config configs/poly/hewl1118_poly.yaml --pipeline-cfg scripts/poly/poly_pipeline_cfg.yaml` |
| 2 | Train | `sbatch scripts/poly/train.slurm` |
| 3 | Predict → careless → refine → peaks | `RUN_DIR=<run-dir> sbatch scripts/poly/pipeline.slurm` |
| 4 | Figures | `python scripts/make_training_figures.py --run-dir <run-dir>` |

Print every command in step 3 without running it:

```bash
python scripts/poly/run_pipeline.py --run-dir <run-dir> --dry-run
python scripts/poly/run_pipeline.py --run-dir <run-dir> --config 1   # a different careless recipe
```

## The careless configs

`careless_configs.py` reproduces `hewl_1118/laue-dials-careless/config{1..6}.sh`
(and the unified `refltorch/scripts/laue_output/careless_scale.sh`), so runs
started here are comparable with everything already scaled on the cluster.
Metadata keys are `BATCH,xcal,ycal,dHKL,wavelength` throughout, which is exactly
what `write_mtz_from_preds` emits.

| config | recipe | here? |
| --- | --- | --- |
| 1 | anomalous, baseline | yes |
| 2 | Friedel split, double Wilson | no — use `careless_scale.sh` |
| 3 | Friedel split, double Wilson, positional encoding | no — use `careless_scale.sh` |
| 4 | anomalous, positional encoding **(default)** | yes |
| 5 | anomalous, image-layers=2, 6k iterations | yes |
| 6 | anomalous, image-layers=2, positional encoding | yes |

Configs 2 and 3 split the data on Friedel mates before scaling and need an
unfriedelize pass afterwards; that path is not ported here.

## The pieces

| File | Role |
| --- | --- |
| `configs/poly/hewl1118_poly.yaml` | Training config: 5-encoder hierarchical, learned-basis profile (latent 16), `polychromatic_wilson` with a degree-40 Chebyshev G(λ) |
| `scripts/poly/preflight.py` | Builds the model, validates the manifest and metadata, prints the real wavelength range, checks every downstream path |
| `scripts/poly/train.slurm` | GPU training with `--figures`; pre-flights first |
| `scripts/poly/careless_configs.py` | The six careless recipes as flag lists |
| `scripts/poly/run_pipeline.py` | The four downstream steps; `--steps`, `--config`, `--dry-run` |
| `scripts/poly/pipeline.slurm` | The driver as a batch job |
| `scripts/poly/poly_pipeline_cfg.yaml` | Envs, careless config, phenix `.eff`s and models |

## Relationship to `refltorch/scripts/laue_output/`

That directory holds the original SLURM-array version of the same pipeline —
`careless_scale.sh`, `submit_scaling.py`, `submit_refinement.py`,
`submit_analysis.py`, `anomalous_peak_heights.py` — which fans out over every
epoch and every config at once. Two things there are on the old contract and
will not work against a run trained with the current integrator:

- they read `run_dir/run_metadata.yaml` with `meta["wandb"]["log_dir"]`; runs
  now write `run_paths.yaml`, which carries `predictions_dir` directly.
- `careless_scale.sh` expects `<epoch_dir>/preds.mtz`; `integrator.predict`
  now writes `preds_epoch_XXXX.mtz`.

`run_pipeline.py` here targets the current contract and runs one config at a
time. Use the refltorch scripts when you want the full fan-out and are willing
to fix those two paths.

## Known gaps in this dataset

`hewl_1118/pytorch_data` predates the `dataset.yaml` manifest, and the data
module raises without one. A manifest reconstructed from `crystal.yaml`,
`stats.pt`, `anscombe_stats.pt`, and the array sizes — 2,664,858 reflections,
1×25×25, anscombe — needs to be copied into the data directory:

```bash
rsync -avz dataset.yaml cannon:/n/lab_storage/hekstra_lab/people/aldama/integrator_data/hewl_1118/pytorch_data/
```

The older `configs/poly/poly_*enc*.yaml` use the pre-refactor data-loader
syntax (`val_split`, `test_split`, `cutoff`, `D`/`H`/`W`, `anscombe`, and a
`stats` key inside `shoebox_file_names`) and raise `TypeError` against the
current `PolychromaticDataModule`. Start from `configs/poly/reference_full.yaml`
or `configs/poly/hewl1118_poly.yaml` instead.
