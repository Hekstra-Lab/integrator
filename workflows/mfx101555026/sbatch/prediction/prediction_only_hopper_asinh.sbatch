#!/bin/bash
#SBATCH --account=lcls:default
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=0-05:00
#SBATCH --job-name=mfx_asinh_ep24_prediction
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/%x_%j.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/%x_%j.err

set -eo pipefail
export PS1="${PS1:-}"

###############################################################################
# Paths
###############################################################################

REPO=/sdf/home/t/thaoh/s3df_practice/integrator
BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx

RUN_DIR="$BASE/runs/run_20260805-011728_9704"
FINAL_CKPT="$RUN_DIR/files/checkpoints/epoch=0024.ckpt"

LOGS_DIR="$BASE/logs"
RUN_DIR_FILE="$LOGS_DIR/prediction_run_dir_${SLURM_JOB_ID}.txt"
TIMING_FILE="$LOGS_DIR/prediction_timing_${SLURM_JOB_ID}.txt"

mkdir -p "$LOGS_DIR"

###############################################################################
# Validate paths
###############################################################################

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: run directory does not exist:"
    echo "  $RUN_DIR"
    exit 1
fi

if [[ ! -f "$FINAL_CKPT" ]]; then
    echo "ERROR: final checkpoint does not exist:"
    echo "  $FINAL_CKPT"
    echo
    echo "Available checkpoints:"
    ls -lh "$RUN_DIR/files/checkpoints"
    exit 1
fi

###############################################################################
# Helper
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
echo "Job ID:                $SLURM_JOB_ID"
echo "Host:                  $(hostname)"
echo "Started:               $JOB_START_TEXT"
echo "Run directory:         $RUN_DIR"
echo "Prediction checkpoint: $FINAL_CKPT"
echo "Allocated CPUs:        ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Python:                $(which python)"
echo "integrator.predict:    $(which integrator.predict)"
echo "============================================================"

python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch CUDA version:", torch.version.cuda)
PY

nvidia-smi

echo "$RUN_DIR" > "$RUN_DIR_FILE"

###############################################################################
# Predict epoch 24 only
###############################################################################

PREDICT_START_EPOCH=$(date +%s)
PREDICT_START_TEXT=$(date --iso-8601=seconds)

echo "============================================================"
echo "Prediction started: $PREDICT_START_TEXT"
echo "Run directory:      $RUN_DIR"
echo "Checkpoint:         $FINAL_CKPT"
echo "============================================================"

integrator.predict \
    --run-dir "$RUN_DIR" \
    --ckpt "$FINAL_CKPT" \
    --batch-size 2048

PREDICT_END_EPOCH=$(date +%s)
PREDICT_END_TEXT=$(date --iso-8601=seconds)
PREDICT_SECONDS=$((PREDICT_END_EPOCH - PREDICT_START_EPOCH))

echo "============================================================"
echo "Prediction finished: $PREDICT_END_TEXT"
echo "Prediction elapsed:  $(format_seconds "$PREDICT_SECONDS")"
echo "============================================================"

###############################################################################
# Timing summary
###############################################################################

JOB_END_EPOCH=$(date +%s)
JOB_END_TEXT=$(date --iso-8601=seconds)
TOTAL_SECONDS=$((JOB_END_EPOCH - JOB_START_EPOCH))

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Run directory: $RUN_DIR"
    echo "Prediction checkpoint: $FINAL_CKPT"

    echo "Job started: $JOB_START_TEXT"
    echo "Prediction started: $PREDICT_START_TEXT"
    echo "Prediction finished: $PREDICT_END_TEXT"
    echo "Job finished: $JOB_END_TEXT"

    echo "Prediction seconds: $PREDICT_SECONDS"
    echo "Prediction elapsed: $(format_seconds "$PREDICT_SECONDS")"

    echo "Total seconds: $TOTAL_SECONDS"
    echo "Total elapsed: $(format_seconds "$TOTAL_SECONDS")"
} | tee "$TIMING_FILE"

echo
echo "Prediction finished successfully."
echo "Timing summary: $TIMING_FILE"
echo "Run directory file: $RUN_DIR_FILE"