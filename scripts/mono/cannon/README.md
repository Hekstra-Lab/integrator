# Cannon working-directory template — monochromatic (rotation) arm

Mirror of `scripts/poly/cannon/` for the DIALS/rotation pipeline. Same front
door and stage layout; the one real difference is the downstream tool (stage 3):
mono scales+merges with **DIALS** and refines with **phenix** (`process_single.py`),
where poly uses **careless**.

## KIT_DIR / OUT split

`env.sh` separates the two roles so a run can never fill the small allocation:

- **KIT_DIR** — these scripts and their configs. Small, version-controlled, on
  `/n/holylabs` (the repo checkout). Configs resolve through here.
- **OUT** — run outputs (checkpoints, predictions, MTZs, figures). Tens of GB,
  on scratch (`$SCRATCH_ROOT/integrator_runs`). Everything a run writes goes here.
  `WB_SAVE_DIR = $OUT/wandb_logs`.

Filling `/n/holylabs` makes every `sbatch` fail at launch (SLURM can't create the
`.out`), which is why outputs must default to scratch.

## Stages

| stage | script | what it does |
|---|---|---|
| preflight | `00_preflight.sh` | validate config + dataset + model build (mode-aware; no GPU) |
| train | `02b_train_requeue.sh` | `integrator.train` on `gpu_requeue`, requeue-safe, resumes from `last.ckpt` |
| pipeline | `03_pipeline.sh` | predict (single ckpt) → DIALS scale+merge → phenix.refine → `rs.find_peaks` → `merging_stats.csv` |
| figures | `04_figures.sh` | per-run training figures + HTML report |
| plots | `05_plots.sh` | multi-model comparison via shared `scripts/make_figures.sh` + `plot_cfg.yaml` |

`03_pipeline.sh` is a single job (not an array): with single-checkpoint predict
there is exactly one refl file, so one `process_single.py` invocation. Predict
needs the GPU; DIALS/phenix run inline on the same node.

## Usage

```bash
./run_all.sh                          # full chain, new run
./run_all.sh --from pipeline --run-name mono_hewl9b7c
./run_all.sh --only plots
./run_all.sh --dry-run                # print the sbatch calls, submit nothing
```

Options: `--run-name`, `--from/--to/--only` (train|pipeline|figures|plots),
`--epochs`, `--config`, `--process-cfg`, `--no-preflight`, `--dry-run`.

## Shared with poly

`00_preflight.sh` calls `scripts/preflight.py` (mode-aware, common), and
`05_plots.sh` calls `scripts/make_figures.sh` with the shared `plot_cfg.yaml`
schema — so one manifest listing mono and poly run dirs produces side-by-side
comparison figures.
