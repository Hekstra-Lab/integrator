#!/bin/bash
#SBATCH --account=lcls:default
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=0-08:00
#SBATCH --job-name=mfx_all_run_chunks_sqrt_squareplus
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/%x_%j.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/%x_%j.err

set -eo pipefail
export PS1="${PS1:-}"

###############################################################################
# Paths
###############################################################################

REPO=/sdf/home/t/thaoh/s3df_practice/integrator
BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx

CONFIG="$REPO/workflows/mfx101555026/configs/mfx_chunked_sqrt_squareplus.yaml"
RUNS_DIR="$BASE/runs"
LOGS_DIR="$BASE/logs"

mkdir -p "$RUNS_DIR" "$LOGS_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file does not exist:"
    echo "  $CONFIG"
    exit 1
fi

BEFORE_FILE="$LOGS_DIR/before_runs_${SLURM_JOB_ID}.txt"
AFTER_FILE="$LOGS_DIR/after_runs_${SLURM_JOB_ID}.txt"
RUN_DIR_FILE="$LOGS_DIR/run_dir_${SLURM_JOB_ID}.txt"
TIMING_FILE="$LOGS_DIR/timing_${SLURM_JOB_ID}.txt"

###############################################################################
# Helper: format seconds as HH:MM:SS
###############################################################################

format_seconds() {
    local total_seconds="$1"

    printf "%02d:%02d:%02d" \
        $((total_seconds / 3600)) \
        $(((total_seconds % 3600) / 60)) \
        $((total_seconds % 60))
}

###############################################################################
# Activate integrator-cuda-dev
###############################################################################

cd "$REPO"

export MAMBA_ROOT_PREFIX=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/micromamba_root

eval "$(
    /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/hewl_1118_tutorial/micromamba/bin/micromamba \
        shell hook --shell bash
)"

micromamba activate integrator-cuda-dev

unset PYTHONHOME
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/src"
export PYTHONUNBUFFERED=1

###############################################################################
# Environment checks
###############################################################################

JOB_START_EPOCH=$(date +%s)
JOB_START_TEXT=$(date --iso-8601=seconds)

echo "============================================================"
echo "Job ID:             $SLURM_JOB_ID"
echo "Host:               $(hostname)"
echo "Started:            $JOB_START_TEXT"
echo "Config:             $CONFIG"
echo "Repository:         $REPO"
echo "Runs directory:     $RUNS_DIR"
echo "Allocated CPUs:     ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Python:             $(which python)"
echo "integrator.train:   $(which integrator.train)"
echo "integrator.predict: $(which integrator.predict)"
echo "============================================================"

python - <<'PY'
import reciprocalspaceship
import torch

print("reciprocalspaceship import: OK")
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)
PY

nvidia-smi

echo
echo "Relevant YAML settings:"
grep -nE \
    "chunk_dir:|batch_size:|num_workers:|max_epochs:|image_level_wilson:|init_log_B:|init_log_G:|n_bins:" \
    "$CONFIG" || true
echo

###############################################################################
# Record existing run directories
###############################################################################

find "$RUNS_DIR" \
    -maxdepth 1 \
    -type d \
    -name "run_*" \
    | sort > "$BEFORE_FILE"

###############################################################################
# Training
###############################################################################

TRAIN_START_EPOCH=$(date +%s)
TRAIN_START_TEXT=$(date --iso-8601=seconds)

echo "============================================================"
echo "Training started: $TRAIN_START_TEXT"
echo "============================================================"

integrator.train \
    --config "$CONFIG" \
    --log-dir "$RUNS_DIR"

TRAIN_END_EPOCH=$(date +%s)
TRAIN_END_TEXT=$(date --iso-8601=seconds)
TRAIN_SECONDS=$((TRAIN_END_EPOCH - TRAIN_START_EPOCH))

echo "============================================================"
echo "Training finished: $TRAIN_END_TEXT"
echo "Training elapsed:  $(format_seconds "$TRAIN_SECONDS")"
echo "============================================================"

###############################################################################
# Find newly created run directory
###############################################################################

find "$RUNS_DIR" \
    -maxdepth 1 \
    -type d \
    -name "run_*" \
    | sort > "$AFTER_FILE"

RUN_DIR=$(comm -13 "$BEFORE_FILE" "$AFTER_FILE" | head -n 1)

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "No new run found by directory comparison."
    echo "Using the newest run_* directory."

    RUN_DIR=$(
        find "$RUNS_DIR" \
            -maxdepth 1 \
            -type d \
            -name "run_*" \
            -printf "%T@ %p\n" \
        | sort -nr \
        | head -n 1 \
        | cut -d' ' -f2-
    )
fi

if [[ -z "${RUN_DIR:-}" || ! -d "$RUN_DIR" ]]; then
    echo "ERROR: could not find the training run directory"
    exit 1
fi

echo "$RUN_DIR" > "$RUN_DIR_FILE"

echo "Run directory: $RUN_DIR"
echo "Run directory saved in: $RUN_DIR_FILE"

###############################################################################
# Prediction
###############################################################################

PREDICT_START_EPOCH=$(date +%s)
PREDICT_START_TEXT=$(date --iso-8601=seconds)

echo "============================================================"
echo "Prediction started: $PREDICT_START_TEXT"
echo "============================================================"

integrator.predict \
    --run-dir "$RUN_DIR" \
    --batch-size 2048

PREDICT_END_EPOCH=$(date +%s)
PREDICT_END_TEXT=$(date --iso-8601=seconds)
PREDICT_SECONDS=$((PREDICT_END_EPOCH - PREDICT_START_EPOCH))

echo "============================================================"
echo "Prediction finished: $PREDICT_END_TEXT"
echo "Prediction elapsed:  $(format_seconds "$PREDICT_SECONDS")"
echo "============================================================"

###############################################################################
# Final timing summary
###############################################################################

JOB_END_EPOCH=$(date +%s)
JOB_END_TEXT=$(date --iso-8601=seconds)
TOTAL_SECONDS=$((JOB_END_EPOCH - JOB_START_EPOCH))

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Config: $CONFIG"
    echo "Run directory: $RUN_DIR"

    echo "Job started: $JOB_START_TEXT"
    echo "Training started: $TRAIN_START_TEXT"
    echo "Training finished: $TRAIN_END_TEXT"
    echo "Prediction started: $PREDICT_START_TEXT"
    echo "Prediction finished: $PREDICT_END_TEXT"
    echo "Job finished: $JOB_END_TEXT"

    echo "Training seconds: $TRAIN_SECONDS"
    echo "Training elapsed: $(format_seconds "$TRAIN_SECONDS")"

    echo "Prediction seconds: $PREDICT_SECONDS"
    echo "Prediction elapsed: $(format_seconds "$PREDICT_SECONDS")"

    echo "Total seconds: $TOTAL_SECONDS"
    echo "Total elapsed: $(format_seconds "$TOTAL_SECONDS")"
} | tee "$TIMING_FILE"

echo
echo "Finished successfully."
echo "Timing summary: $TIMING_FILE"
echo "Run directory file: $RUN_DIR_FILE"