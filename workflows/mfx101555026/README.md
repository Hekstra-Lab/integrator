# MFX101555026 — Integrator Workflow

End-to-end workflow for applying the variational-inference Integrator to
experimental MFX (Macromolecular Femtosecond Crystallography) data at LCLS,
then evaluating downstream crystallographic performance against the
cctbx.xfel baseline.

---

## What this branch adds

Extensions built on top of the base Integrator framework (Luis Aldama):

| Extension | Description |
|---|---|
| **MFX/Jungfrau shoebox extraction** | `make_shoeboxes --mfx`: reads cctbx.xfel `.refl/.expt` pairs, handles Jungfrau 32-panel XTC format, clips shoeboxes at panel boundaries, applies detector mask. Parallel extraction via 16-job Slurm array. |
| **Chunked preprocessing** | `integrator.preprocess`: two-pass CLI that filters, splits, transforms, and writes fixed-size chunks from a raw shoebox dataset. Enables training on datasets too large for RAM. Produces `manifest.yaml` (version 2) with deterministic train/val split. |
| **ChunkedDataModule** | `chunked_rotation_data` loader: IterableDataset that loads one chunk at a time, yields complete mini-batches, shuffles at chunk and row level, preserves row order during prediction for write-back. |
| **LogNormal / FoldedNormal qi** | Two new intensity posteriors (`lognormal`, `foldednormal`) registered as surrogate options. FoldedNormal required a custom `Distribution` subclass to provide `log_prob` and `mean`. |
| **MC-KL fallback** | `_kl_divergence_with_mc_fallback`: automatically falls back to Monte Carlo KL estimation when PyTorch lacks an exact formula (e.g. FoldedNormal vs. Gamma Wilson prior). |
| **Normal / Student-t likelihoods** | `observation_likelihood: normal` or `student_t` in the Wilson loss. Added to support MFX/Jungfrau pixel values that can be floating-point and negative. |
| **MFX write-back** | `refl_io.py`: maps predicted I and σ(I) back to the original 124,975 `.refl` files via `refl_ids` + `reflection_id` metadata. Vectorised `np.unique` groupby (was O(N × n_files), now O(N log N)). Float32 clamping fix for refl_ids > 2²³. |
| **Image-level Wilson B/G** *(prototype)* | `image_level_wilson: true` in the loss: per-image B and G via `nn.Embedding` tables indexed by `image_id`. Implemented and tested; not yet evaluated downstream. |

---

## Dataset

| Property | Value |
|---|---|
| Experiment | mfx101555026 |
| Source files | 124,975 indexed MFX `.refl/.expt` pairs |
| Runs used | r0269–r0289 (excluding r0275) |
| Shoeboxes extracted | ~115 million reflections |
| Shoebox size | 1 × 25 × 25 pixels |
| Detector | Jungfrau (32 panels, XTC format) |

---

## Pipeline

```
MFX raw data  (cctbx.xfel indexed + integrated)
  r*_idx-data_*_integrated.refl / .expt
        │
        ▼  [cctbx env]
  make_shoeboxes  (MFX/Jungfrau path)
        │
        ▼
  counts.npy  masks.npy  metadata.npy  dataset.yaml
        │
        ▼  [integrator-cuda-dev]
  integrator.preprocess
  (filter → split → transform → write chunks)
        │
        ▼
  chunk_00000/ … chunk_NNNNN/
  manifest.yaml  (version 2)
        │
        ▼  [integrator-cuda-dev, GPU]
  integrator.train  (ChunkedDataModule + Wilson ELBO)
        │
        ▼
  run_*/files/checkpoints/epoch=XXXX.ckpt
        │
        ▼  [integrator-cuda-dev, GPU]
  integrator.predict
        │
        ▼
  epoch_XXXX/*.parquet  (qi_mean, qi_var, qbg_mean, refl_ids, …)
        │
        ▼  [integrator-cuda-dev]
  export_preds_for_dials.py  (parquet → .npz)
        │
        ▼  [cctbx env]
  run_writeback.py  (dials.python)
        │
        ▼
  mfx_refl_writeback/  (124,975 .refl + .expt)
        │
        ▼  [cctbx env]
  cxi.xfel.merge  (scale PHIL)
  cxi.xfel.merge  (merge PHIL)
        │
        ▼
  merged MTZ  +  CC½ / Rint / Rsplit diagnostics
        │
        ▼  [cctbx env]
  phenix.refine
        │
        ▼
  R-work / R-free  +  electron density maps
```

---

## Environment rule

Three environments — never mix them.

| Environment | Activate | Use for |
|---|---|---|
| **cctbx / psana2** | `source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh` | make_shoeboxes, write-back, scale, merge, refine |
| **integrator-cuda-dev** | `micromamba activate integrator-cuda-dev` | preprocess, train, predict, NPZ export |
| **CCP4 / visualization** | `source /sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh` | COOT structure and map viewing |

> **Note:** `integrator-cuda-dev` has no DIALS/dxtbx. Do not run write-back there.
> `cctbx` has no polars/torch. Do not run training there.

---

## Step-by-step walkthrough

Set paths once at the top of your session:

```bash
REPO=/sdf/home/t/thaoh/s3df_practice/integrator
BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
DATA_DIR=$BASE/mfx_shoebox_allruns_269_289_no275_024_rg101
CHUNK_DIR=$DATA_DIR/chunks
REFL_DIR=$BASE/all_runs_269_289_no275_024_rg101_integrated
MASK=/sdf/data/lcls/ds/mfx/mfx101555026/results/pam/hot_lines_combined5.mask
```

---

### Step 1 — Extract shoeboxes  *(cctbx env)*

Skip if `DATA_DIR` already contains `counts.npy`, `masks.npy`, `metadata.npy`,
and `dataset.yaml`.

```bash
source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=$REPO/src:$PYTHONPATH

dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir "$REFL_DIR" \
  --out-dir  "$DATA_DIR" \
  --mfx-pattern "r*_idx-data_*_integrated.refl" \
  --detector-mask "$MASK" \
  --w 25 --h 25 --d 1 \
  --counts-dtype float32 \
  --write-chunk-size 10000 \
  --no-mask-overlap
```

For the full dataset (124,975 files) use the 16-job Slurm array script:

```bash
sbatch $REPO/workflows/mfx101555026/scripts/extraction/make_shoeboxes_allruns_024_rg101_16array.sh
# then merge chunks:
python $REPO/workflows/mfx101555026/scripts/extraction/merge_allruns_024_rg101_shoebox_chunks.py
```

Expected output under `DATA_DIR/`:
```
counts.npy    masks.npy    metadata.npy    dataset.yaml
```

---

### Step 2 — Activate the integrator environment

```bash
export MAMBA_ROOT_PREFIX=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/micromamba_root
eval "$(/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/hewl_1118_tutorial/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate integrator-cuda-dev
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=$REPO/src
cd $REPO
```

Verify:
```bash
which integrator.preprocess integrator.train integrator.predict
# if missing: pip install -e .
```

---

### Step 3 — Preprocess into chunks  *(integrator-cuda-dev)*

Run once per dataset + preprocessing configuration.

```bash
integrator.preprocess \
  --data-dir "$DATA_DIR" \
  --validation-split 0.2 \
  --split-seed 42 \
  --transform asinh \
  --chunk-size 50000 \
  --n-bins 1 \
  --out-dir "$CHUNK_DIR" \
  -v
```

> `--n-bins` must match `loss.args.n_bins` in the training YAML.
> `--transform` must match the transform expected by the config.

Verify `manifest.yaml`:
```bash
head -40 "$CHUNK_DIR/manifest.yaml"
# expect: version: 2, transform: asinh, n_bins: 1, source_data_dir: $DATA_DIR
```

For the `sqrt_squareplus` transform variant:
```bash
integrator.preprocess \
  --data-dir "$DATA_DIR" \
  --validation-split 0.2 \
  --split-seed 42 \
  --transform sqrt_squareplus \
  --chunk-size 50000 \
  --n-bins 1 \
  --out-dir "$DATA_DIR/chunks_sqrt_squareplus" \
  -v
```

---

### Step 4 — Request a GPU node

```bash
srun \
  --account=lcls:prjlumine22 \
  --partition=ampere \
  --qos=normal \
  --constraint=OS_VER:8.6 \
  --nodes=1 \
  --cpus-per-task=24 \
  --gpus=1 \
  --mem=256G \
  --time=0-03:00 \
  --job-name=mfx_chunked \
  --pty bash -i
```

After entering the node, repeat Step 2 to activate `integrator-cuda-dev`. Verify:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

### Step 5 — Train  *(integrator-cuda-dev, GPU)*

**Interactive:**
```bash
RUNS_DIR=$BASE/runs
CONFIG=$REPO/workflows/mfx101555026/configs/mfx_chunked_5000.yaml

/usr/bin/time -v integrator.train \
  --config  "$CONFIG" \
  --log-dir "$RUNS_DIR" \
  2>&1 | tee "$BASE/logs/train_$(date +%Y%m%d-%H%M%S).log"

# Find the new run directory:
RUN_DIR=$(find "$RUNS_DIR" -maxdepth 1 -type d -name 'run_*' \
          -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
echo "$RUN_DIR"
```

**Batch (Slurm):** edit paths in the sbatch script, then submit:
```bash
sbatch $REPO/workflows/mfx101555026/sbatch/training/auto_train_predict_chunked.sbatch
# Hopper (preemptable) variant:
sbatch $REPO/workflows/mfx101555026/sbatch/training/auto_train_predict_chunked_HOPPER.sbatch
```

Checkpoints land in `$RUN_DIR/files/checkpoints/`.

---

### Step 6 — Predict  *(integrator-cuda-dev, GPU)*

```bash
FINAL_CKPT="$RUN_DIR/files/checkpoints/epoch=0024.ckpt"

/usr/bin/time -v integrator.predict \
  --run-dir    "$RUN_DIR" \
  --ckpt       "$FINAL_CKPT" \
  --batch-size 2048 \
  2>&1 | tee "$BASE/logs/predict_$(date +%Y%m%d-%H%M%S).log"
```

Output: `$RUN_DIR/files/predictions/epoch_0024/` (~5,758 parquet files).
Estimated time: ~1.5 hours on Ampere A100.

---

### Step 7a — Export predictions to NPZ  *(integrator-cuda-dev)*

Converts parquet predictions to `.npz` for the cctbx write-back environment
(cctbx has no polars; `.npz` is readable everywhere).

```bash
PRED_DIR="$RUN_DIR/files/predictions/epoch_0024"
PRED_NPZ="$RUN_DIR/files/predictions/preds_epoch_0024.npz"

python $REPO/workflows/mfx101555026/scripts/writeback/export_preds_for_dials.py \
  --ckpt-dir "$PRED_DIR" \
  --out       "$PRED_NPZ"
# ~29 seconds on Ampere; ~115M rows, ~3.5 GB
```

**Batch (Slurm):**
```bash
# edit paths inside the script first, then:
sbatch $REPO/workflows/mfx101555026/sbatch/writeback/asinh/export_asinh_ep24_npz.sbatch
sbatch $REPO/workflows/mfx101555026/sbatch/writeback/sqrt/export_sqrt_ep24_npz.sbatch
```

---

### Step 7b — Write back to .refl files  *(cctbx env)*

```bash
source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=$REPO/src:$PYTHONPATH

WB_OUT="$RUN_DIR/files/predictions/mfx_refl_writeback"

dials.python $REPO/workflows/mfx101555026/scripts/writeback/run_writeback.py \
  --npz      "$PRED_NPZ" \
  --metadata "$DATA_DIR/metadata.npy" \
  --refl-dir "$REFL_DIR" \
  --out-dir  "$WB_OUT"
# ~17 minutes for 124,975 files + 115M prediction rows
```

Verify:
```bash
ls "$WB_OUT" | wc -l    # ~249,950  (refl + expt symlink pairs)
```

**Batch (Slurm):**
```bash
sbatch $REPO/workflows/mfx101555026/sbatch/writeback/asinh/writeback_asinh_ep24.sbatch
sbatch $REPO/workflows/mfx101555026/sbatch/writeback/sqrt/writeback_sqrt_ep24.sbatch
```

#### Smoke test before a full run

```bash
# Step 1 — integrator-cuda-dev
python $REPO/workflows/mfx101555026/scripts/writeback/test_predict_small.py \
  --config   "$CONFIG" \
  --ckpt     "$FINAL_CKPT" \
  --metadata "$DATA_DIR/metadata.npy" \
  --out-dir  /tmp/test_smoke \
  --n-images 5

# Step 2 — dials.python
dials.python $REPO/workflows/mfx101555026/scripts/writeback/test_writeback_small.py \
  --npz      /tmp/test_smoke/preds_small.npz \
  --metadata "$DATA_DIR/metadata.npy" \
  --refl-dir "$REFL_DIR" \
  --out-dir  /tmp/test_smoke/writeback
# Expected: ALL CHECKS PASSED
```

---

### Step 8 — Scale and merge  *(cctbx env)*

Both scaling and merging use `cxi.xfel.merge` — the command is the same,
the PHIL file determines which operation runs.

PHIL files are in `$REPO/workflows/mfx101555026/combinedmask_scale_merge_phil/`.

```bash
PHIL_DIR=$REPO/workflows/mfx101555026/combinedmask_scale_merge_phil

# Step 1 — scale (run the scaling PHIL with cxi.xfel.merge):
cxi.xfel.merge $PHIL_DIR/scale_matched_cctbx_asinh_dmin2.phil

# Step 2 — merge (run the merge PHIL with cxi.xfel.merge):
cxi.xfel.merge $PHIL_DIR/merge_matched_cctbx_asinh_dmin2.phil
```

For the Integrator (asinh / sqrt-squareplus) variants:
```bash
cxi.xfel.merge $PHIL_DIR/PREDICTED_scale_all_r269_r289__dmin2_ASINH.phil
cxi.xfel.merge $PHIL_DIR/PREDICTED_merge_all_r269_r289__dmin2_ASINH.phil

cxi.xfel.merge $PHIL_DIR/PREDICTED_scale_all_r269_r289__dmin2_SQRT.phil
cxi.xfel.merge $PHIL_DIR/PREDICTED_merge_all_r269_r289__dmin2_SQRT.phil
```

**Batch (Slurm):**
```bash
sbatch $REPO/workflows/mfx101555026/sbatch/scale_merge/run_scale_merge_matched_cctbx_asinh_dmin2.sbatch
sbatch $REPO/workflows/mfx101555026/sbatch/scale_merge/run_scale_merge_matched_cctbx_sqrt_dmin2.sbatch
```

For matched comparison (build the same-image subset used by both cctbx and Integrator):
```bash
sbatch $REPO/workflows/mfx101555026/sbatch/matching/match_cctbx_asinh.sbatch
sbatch $REPO/workflows/mfx101555026/sbatch/matching/match_cctbx_sqrt.sbatch
```

---

### Step 9 — PHENIX refinement  *(cctbx env)*

```bash
BASE_CCTBX=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
REF_PDB=/sdf/data/lcls/ds/mfx/mfx101555026/results/offline/datasets/\
ClCRY4_wt_1us_74mJcm2_ellipse/processing/v01_new_defaults/diffmaps/dimple_dark/final.pdb

phenix.refine \
  "$BASE_CCTBX/phenix_results/asinh_dmin2/input.mtz" \
  "$REF_PDB" \
  refinement.main.number_of_macro_cycles=5
```

---

## Training configs

| Config | Transform | qi posterior | Notes |
|---|---|---|---|
| `mfx_chunked_5000.yaml` | asinh | FoldedNormal | Production run (asinh) |
| `mfx_chunked_sqrt_squareplus.yaml` | sqrt_squareplus | FoldedNormal | Production run (sqrt) |
| `mfx_chunked_test10.yaml` | asinh | FoldedNormal | Smoke test (10 files) |
| `mfx_test_10_files.yaml` | asinh | Gamma | Legacy small test |

All chunked configs use `data_loader.name: chunked_rotation_data` and
`loss.args.image_level_wilson: false` (global B/G).

Key loss settings:
```yaml
loss:
  name: monochromatic_wilson
  args:
    observation_likelihood: normal   # MFX/Jungfrau pixels can be float/negative
    init_log_B: 20.0                 # softplus → B ≈ 20
    init_log_G: 1000.0               # softplus → G ≈ 1000
    image_level_wilson: false
    n_bins: 1
    lp_correction: false
```

---

## Results

### Matched merging statistics (apples-to-apples comparison)

*Both cctbx and Integrator merged from the same image subset.*

| Method | CC½ | Rint | Rsplit | Experiments | Reflections |
|---|---|---|---|---|---|
| cctbx matched (asinh subset) | 0.998 | 6.9% | 5.2% | 113,420 | 36,326,432 |
| Integrator — asinh | 0.990 | 15.8% | 11.1% | 113,804 | 32,066,173 |
| cctbx matched (sqrt subset) | 0.998 | 7.0% | 5.3% | 113,525 | 36,351,064 |
| Integrator — sqrt-squareplus | 0.990 | 15.8% | 11.1% | 113,910 | 32,094,591 |

### PHENIX refinement (5 macro-cycles, 2.0 Å)

| Method | R-work | R-free | N reflections |
|---|---|---|---|
| cctbx matched (asinh subset) | 0.179 | 0.213 | 44,236 |
| Integrator — asinh | 0.261 | 0.277 | 45,923 |
| cctbx matched (sqrt subset) | 0.179 | 0.214 | 44,276 |
| Integrator — sqrt-squareplus | 0.262 | 0.275 | 45,891 |

### CC½ per resolution bin (selected bins)

| Bin | d_min (Å) | cctbx (matched) | Integrator asinh | Integrator sqrt |
|---|---|---|---|---|
| 1 | 5.43 | 0.999 | 0.993 | 0.993 |
| 7 | 2.84 | 0.978 | 0.675 | 0.657 |
| 11 | 2.44 | 0.793 | 0.512 | 0.538 |
| 17 | 2.11 | 0.128 | 0.545 | 0.574 |
| 20 | 2.00 | 0.055 | 0.473 | 0.551 |

Full 20-bin table: see `refinement_comparison.txt` (local, not committed).

**Key observation:** cctbx CC½ drops steeply above ~2.4 Å. Both Integrator
variants remain elevated at high resolution — behavior that requires further
validation before interpreting as recovered signal.

---

## Workflow files reference

```
workflows/mfx101555026/
  configs/
    mfx_chunked_5000.yaml                          production config — asinh + FoldedNormal
    mfx_chunked_sqrt_squareplus.yaml               production config — sqrt_squareplus + FoldedNormal
    mfx_chunked_test10.yaml                        smoke-test config (10 files)
    mfx_test_10_files.yaml                         legacy small-test (rotation_data loader)
    mfx_allruns_asinh_foldednormal_imagewilson_global_B_G.yaml   image-level Wilson variant
    mfx_allruns_sqrt_squareplus_foldednormal_imagewilson_global_B_G.yaml
    (+ other experimental configs)

  combinedmask_scale_merge_phil/
    scale_matched_cctbx_asinh_dmin2.phil           scale — matched cctbx (asinh subset)
    merge_matched_cctbx_asinh_dmin2.phil           merge — matched cctbx (asinh subset)
    scale_matched_cctbx_sqrt_dmin2.phil            scale — matched cctbx (sqrt subset)
    merge_matched_cctbx_sqrt_dmin2.phil            merge — matched cctbx (sqrt subset)
    PREDICTED_scale_all_r269_r289__dmin2_ASINH.phil  scale — Integrator asinh predictions
    PREDICTED_merge_all_r269_r289__dmin2_ASINH.phil  merge — Integrator asinh predictions
    PREDICTED_scale_all_r269_r289__dmin2_SQRT.phil   scale — Integrator sqrt predictions
    PREDICTED_merge_all_r269_r289__dmin2_SQRT.phil   merge — Integrator sqrt predictions
    scale_all_r269_r289__dmin2.phil                scale — full cctbx allruns
    merge_all_r269_r289__dmin2.phil                merge — full cctbx allruns
    (+ older d_min variants: dmin15, dmin21, dmin26, dmin28)

  sbatch/
    training/
      auto_preprocess.sbatch                       preprocess dataset into chunks
      auto_train_predict_chunked.sbatch            train + predict on Ampere (sequential)
      auto_train_predict_chunked_HOPPER.sbatch     train + predict on Hopper (preemptable)
      resume_training.batch                        resume a paused training run
      resume_training_HOPPER.batch                 resume on Hopper
    prediction/
      prediction_only_ampere_sqrt.sbatch           predict only — sqrt run, Ampere
      prediction_only_hopper_asinh.sbatch          predict only — asinh run, Hopper
    writeback/
      asinh/
        export_asinh_ep24_npz.sbatch               parquet → .npz for asinh epoch 24
        writeback_asinh_ep24.sbatch                .npz → .refl write-back for asinh
      sqrt/
        export_sqrt_ep24_npz.sbatch                parquet → .npz for sqrt epoch 24
        writeback_sqrt_ep24.sbatch                 .npz → .refl write-back for sqrt
    scale_merge/
      run_scale_merge_matched_cctbx_asinh_dmin2.sbatch
      run_scale_merge_matched_cctbx_sqrt_dmin2.sbatch
      run_scale_merge_all_dmin2_ASINH.sbatch
      run_scale_merge_all_dmin2_SQRT.sbatch
    matching/
      match_cctbx_asinh.sbatch                     build same-image subset (asinh)
      match_cctbx_sqrt.sbatch                      build same-image subset (sqrt)

  scripts/
    extraction/
      make_shoeboxes_allruns_024_rg101_16array.sh  16-job Slurm array for shoebox extraction
      merge_allruns_024_rg101_shoebox_chunks.py    merge chunk outputs into final dataset
      merge_allruns_024_rg101_shoebox_chunks.sbatch

    writeback/
      export_preds_for_dials.py        parquet → .npz  (integrator-cuda-dev)
      run_writeback.py                 .npz → .refl write-back  (dials.python, production)
      test_predict_small.py            smoke test: prediction on 5 files
      test_writeback_small.py          smoke test: write-back + 4 validation checks
      test_predict_writeback_small.py  end-to-end smoke test
      diagnose_refl_ids.py             debug refl_id mismatches / float32 precision

    diagnostics/
      inspect_mfx_shoeboxes.py         visualise extracted shoeboxes
      inspect_qp_mean_profiles.py      inspect learned spot profile qp.mean
      inspect_cctbx_refl_expt.py       inspect cctbx .refl/.expt structure
      inspect_integrator_parquet.py    inspect prediction parquet contents
      plot_loss_history.py             plot training loss curves
      plot_cc12_vs_resolution.py       CC½ vs resolution (cctbx vs Integrator)
      plot_cctbx_selected_profiles.py  plot cctbx selected spot profiles
      plot_mfx_negative_pixels.py      visualise negative pixel distribution
      plot_qi_by_strength.py           qi distribution + mean profiles stratified by
                                       weak/medium/strong (cctbx intensity percentiles
                                       p10 / p50 / p90)

    archive/                           old scripts kept for reference; not part of
                                       the active workflow
```

---

## Data and scratch paths

```
BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx

$BASE/
  mfx_shoebox_allruns_269_289_no275_024_rg101/       shoebox dataset
    counts.npy  masks.npy  metadata.npy  dataset.yaml
    chunks/                                           preprocessed chunks
      manifest.yaml
      chunk_00000/ … chunk_NNNNN/
  all_runs_269_289_no275_024_rg101_integrated/        original .refl/.expt files
  runs/                                               training run directories
    run_20260805-011728_9704/                         asinh run (epoch 0-24)
    run_20260805-183129_a8b9/                         sqrt_squareplus run (epoch 0-24)
  phenix_results/
    matched_cctbx_asinh_dmin2/2026_refine_001.{mtz,cif}
    matched_cctbx_sqrt_dmin2/2026_refine_001.{mtz,cif}
    asinh_dmin2/2026_refine_001.{mtz,cif}
    sqrt_dmin2/2026_refine_001.{mtz,cif}

$REPO/workflows/mfx101555026/combinedmask_scale_merge_phil/
  PHIL files for cxi.xfel.merge — scale and merge PHILs for matched and Integrator variants
```

> Large data (shoeboxes, training runs, prediction parquets, write-back .refl
> files) lives in scratch and is **not** committed to Git.

---

## Viewing the structure in COOT

```bash
# SSH to sdfiana003 with X11 forwarding:
ssh -X -J s3dflogin.slac.stanford.edu thaoh@sdfiana003

# Source CCP4:
source /sdf/group/lcls/ds/tools/ccp4-8.0/bin/ccp4.setup-sh

# Open matched cctbx refinement result:
coot \
  --pdb  $BASE/phenix_results/matched_cctbx_asinh_dmin2/2026_refine_001.cif \
  --auto $BASE/phenix_results/matched_cctbx_asinh_dmin2/2026_refine_001.mtz
```

Use NoMachine (`https://s3dfnx.slac.stanford.edu:4443/`) for smoother 3D rendering.
