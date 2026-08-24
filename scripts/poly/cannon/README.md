# Cannon working-directory template

The orchestration layer for a Laue run on FASRC `cannon`: thin wrappers around
`scripts/poly/` plus the site-specific settings (`env.sh`) and the two
downstream configs.

**Copy this directory before using it**, then edit `env.sh`. The scripts and
their configs stay where they are copied (`KIT_DIR`, small, versioned); run
outputs go to `OUT` on scratch, which defaults to
`/n/netscratch/hekstra_lab/Lab/laldama/integrator_runs`.

That split is not cosmetic. A single run writes tens of GB of checkpoints,
predictions and MTZs, and `/n/holylabs` is a small per-lab allocation:
filling it makes every `sbatch` fail at launch, because SLURM cannot create
the job's `.out` file. Outputs must never default to the script directory.

```bash
cp -r $INTEGRATOR_ROOT/scripts/poly/cannon ~/my_laue_run
cd ~/my_laue_run
./run_all.sh --dry-run
```

`env.sh` is the one file to edit per site: repo checkout, micromamba hook,
environment names, and the W&B project. Everything else sources it.

The single entry point is `run_all.sh`, which submits every stage as its own
SLURM job chained with `--dependency=afterok`. The numbered scripts remain
useful for running one stage by hand.

Verified end to end on 2026-08-23 against `hewl_1118`: train (40 epochs) →
predict → careless config4 → phenix.refine ×2 → `rs.find_peaks` +
`anomalous_peak_heights.py` + `careless.ccanom`.

---

# HEWL Laue run — working directory

Everything here is a thin wrapper around the repo at `$INTEGRATOR_ROOT`
(`/n/lab_storage/hekstra_lab/people/aldama/software/integrator`), so the
scripts never drift from the code. Edit `env.sh` to point somewhere else.

Run dirs land in this directory.

## Order

```bash
./00_preflight.sh                      # seconds, login node, no GPU
./01_smoke.sh                          # 2 epochs on 50k refl — proves the chain
PIPELINE_CFG=$PWD/poly_pipeline_smoke_cfg.yaml ./03_pipeline.sh smoke   # careless 300 iters
sbatch 02b_train_requeue.sh            # the real run, scavenger partition (see below)
./03_pipeline.sh <run-dir> --dry-run   # eyeball the four commands
./03_pipeline.sh <run-dir>             # submit them
./04_figures.sh <run-dir>              # figures + one HTML page
```

The smoke pass is worth the twenty minutes: it exercises predict → careless →
phenix → peaks on real data, so a wrong `.eff` or a missing column fails
before the 12-hour job rather than after it.

## Files

| File | What it is |
| --- | --- |
| `env.sh` | Repo root, micromamba envs, run root. Everything else sources this. |
| `00_preflight.sh` | Builds the model, checks the manifest, metadata columns, and every downstream path |
| `01_smoke.sh` | 2 epochs on 50k reflections, 2h wall limit |
| `02_train.sh` | Full run on `-p gpu`; `EPOCHS=`, `RUN_NAME=`, `WB_PROJECT=` to override |
| `02b_train_requeue.sh` | Same run on `-p gpu_requeue`, preemption-safe. `sbatch` it directly. |
| `03_pipeline.sh` | predict → careless config4 → phenix ×2 → peaks; `--dry-run`, `STEPS=` |
| `04_figures.sh` | Per-run training figures + `report.html` |
| `05_plots.sh` | Multi-run comparison: loss, model-vs-DIALS, peaks, R-factors |
| `plot_cfg.yaml` | Which runs to compare, and the reference data |
| `hewl1118_smoke.yaml` | The training config with `subset_size: 50000`, 2 epochs |
| `poly_pipeline_cfg.yaml` | Envs, careless config, phenix effs and models |
| `poly_pipeline_smoke_cfg.yaml` | Same, with careless at 300 iterations |

## Fairshare and the scavenger partition

`gpu` has 36 nodes and is fairshare-gated; as of 2026-08-23 the account's
fairshare sits at 0.014, so queueing there means waiting. `gpu_requeue` has
436 nodes on the same hardware, is not gated the same way, and had idle
capacity — use `02b_train_requeue.sh` and the job starts.

The trade is preemption. It is safe here: the run checkpoints every 2 epochs,
SLURM restarts the script with the same job id so the run dir is stable, and
the script resumes from `last.ckpt`. Worst case a preemption costs 2 epochs.

`gpu_test` is 12 nodes x 8 A100 3g.20gb MIG slices (96 slices, 12h cap) and
usually has the shortest wait. It rejects any job asking 8 or more cores per
GPU, so use the wrapper rather than sbatching 02b there directly:

```bash
./02c_train_gputest.sh          # -c 7, 6 workers, 40 epochs
```

At roughly 7 min/epoch on one MIG slice, 40 epochs is ~5h and 100 would not
fit in the 12h cap. Resubmit the same command to continue: `RUN_NAME` is
pinned, so it picks up from `last.ckpt`.

## Weights & Biases

On by default, project `hewl_laue` (`WB_PROJECT` in `env.sh`). Everything also
lands locally regardless — `plots/metrics.csv`, the loss curves, the figure
dumps. To go local-only for one job:

```bash
WB_PROJECT= ./02c_train_gputest.sh
```

Separate the smoke from the real runs if you like: `WB_PROJECT=hewl_laue_smoke
./01_smoke.sh`.

Two things the scripts handle for you:

- W&B moves the output root under `<WB_SAVE_DIR>/wandb/run-<id>/`, so
  checkpoints no longer live in `<run-dir>/files`. `02b` reads `log_dir` out
  of `run_paths.yaml` instead of assuming a layout, and every downstream
  script does the same.
- A requeued job would otherwise open a *new* W&B run each time and split the
  curves. `02b` passes `--wandb-resume-id` with the id recorded in
  `run_paths.yaml`, so every attempt logs into one run.

## Partition limits on FASRC cannon

`gpu_test` allows **2 submitted jobs per user**, running or pending, and caps
a user at 8 GPUs / 64 CPUs / 512 GB across them. A third submission is
rejected with `QOSMaxSubmitJobPerUserLimit`, which surfaces as an
instant-failure job with no output file rather than as a queued job.

The limit is per *user*, so anything else running under the same account
counts against it — including another session's jobs. Before queuing a
`gpu_test` chain, check what is already there:

```bash
squeue -u $USER -p gpu_test
```

Per-GPU limits on that partition, enforced by a submit filter: fewer than 8
cores and under 65,536 MB per GPU. `02c_train_gputest.sh` asks for `-c 7`
and `--mem=60G` for that reason, and `03_pipeline.sh` matches it.

`gpu_requeue` has no such submit cap and a far larger node pool, at the cost
of preemption. `02b_train_requeue.sh` resumes from `last.ckpt`, so a
preemption there costs at most one checkpoint interval.

## What to check while it runs

In the first minutes of the training job:

- `Figure dumps: <output_root>/figures` — `--figures` took effect
- `tracking 12 shoeboxes: weak:… medium:… strong:…` at the first validation
  epoch — the tracked set was chosen

After careless:

- `config4_0.mtz` and `config4_xval_0.mtz` under `<epoch>/scaling/`
- peaks land in `<epoch>/scaling/config4_refine/peaks.csv` (rs.find_peaks
  schema, read by `plot_peaks.py`), alongside `peak_heights.csv` and
  `ccanom.csv`/`.png`

## Settings baked in

Verified live on 2026-08-23:

- envs `integrator-cuda-dev` (train, predict, figures) and `crls` (careless)
- phenix 1.20 garden at `/n/hekstra_lab_tier0/...`, effs and models on `/n/holylabs`
- λ range 0.9700–1.2500 Å, matching the Chebyshev domain in the config
- 992 images → `BATCH`, so careless can scale per image

## Comparison baseline

`hewl_1118/laue-dials-careless/config3/` holds a complete laue-dials → careless
→ refine → peaks run on the same crystal with the same refinement effs. Its
`peaks.csv` is the like-for-like number to beat.
