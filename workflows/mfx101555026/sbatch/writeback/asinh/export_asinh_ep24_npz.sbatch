#!/bin/bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=ampere
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0-05:00
#SBATCH --job-name=mfx_asinh_ep24_export_npz
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

PRED_DIR="$RUN_DIR/predictions/epoch_0024"
PRED_NPZ="$RUN_DIR/predictions/preds_epoch_0024.npz"

LOGS_DIR="$BASE/logs"
mkdir -p "$LOGS_DIR"

###############################################################################
# Validate prediction shards
###############################################################################

if [[ ! -d "$PRED_DIR" ]]; then
    echo "ERROR: prediction directory does not exist:"
    echo "  $PRED_DIR"
    exit 1
fi

N_PARQUET=$(find "$PRED_DIR" -maxdepth 1 -type f -name '*.parquet' | wc -l)

if [[ "$N_PARQUET" -ne 5758 ]]; then
    echo "ERROR: expected 5758 prediction parquet files, found $N_PARQUET"
    echo "Prediction directory:"
    echo "  $PRED_DIR"
    exit 1
fi

echo "Prediction set complete: $N_PARQUET parquet files."

###############################################################################
# Activate integrator-cuda-dev
###############################################################################

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

cd "$REPO"

###############################################################################
# Environment information
###############################################################################

echo "============================================================"
echo "MFX ASINH epoch-24 NPZ export"
echo "Job ID:        $SLURM_JOB_ID"
echo "Host:          $(hostname)"
echo "Started:       $(date --iso-8601=seconds)"
echo "Run:           $RUN_DIR"
echo "Prediction dir:$PRED_DIR"
echo "Parquet files: $N_PARQUET"
echo "Output NPZ:    $PRED_NPZ"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory:        128G"
echo "Python:        $(which python)"
echo "============================================================"

###############################################################################
# Export all epoch-24 prediction shards -> NPZ
###############################################################################

START=$(date +%s)

/usr/bin/time -v \
python "$REPO/workflows/mfx101555026/scripts/writeback/export_preds_for_dials.py" \
    --ckpt-dir "$PRED_DIR" \
    --out "$PRED_NPZ"

END=$(date +%s)
ELAPSED=$((END - START))

###############################################################################
# Verify output
###############################################################################

if [[ ! -f "$PRED_NPZ" ]]; then
    echo "ERROR: expected NPZ was not created:"
    echo "  $PRED_NPZ"
    exit 1
fi

echo
echo "============================================================"
echo "Export completed successfully."
echo "Finished: $(date --iso-8601=seconds)"

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))

echo
echo "Output:"
ls -lh "$PRED_NPZ"

echo "============================================================"