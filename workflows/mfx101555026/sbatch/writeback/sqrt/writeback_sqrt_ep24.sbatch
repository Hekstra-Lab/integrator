#!/bin/bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=ampere
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=0-02:00
#SBATCH --job-name=mfx_sqrt_ep24_writeback
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/%x_%j.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/%x_%j.err

set -eo pipefail
export PS1="${PS1:-}"

###############################################################################
# Paths
###############################################################################

REPO=/sdf/home/t/thaoh/s3df_practice/integrator
BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx

DATA_DIR="$BASE/mfx_shoebox_allruns_269_289_no275_024_rg101"
RUN_DIR="$BASE/runs/run_20260805-183129_a8b9"
REFL_DIR="$BASE/all_runs_269_289_no275_024_rg101_integrated"

PRED_NPZ="$RUN_DIR/predictions/preds_epoch_0024.npz"
WB_OUT="$RUN_DIR/predictions/mfx_refl_writeback"

LOGS_DIR="$BASE/logs"
mkdir -p "$LOGS_DIR"

###############################################################################
# Validate inputs
###############################################################################

if [[ ! -f "$PRED_NPZ" ]]; then
    echo "ERROR: NPZ prediction file does not exist:"
    echo "  $PRED_NPZ"
    exit 1
fi

if [[ ! -f "$DATA_DIR/metadata.npy" ]]; then
    echo "ERROR: metadata.npy does not exist:"
    echo "  $DATA_DIR/metadata.npy"
    exit 1
fi

if [[ ! -d "$REFL_DIR" ]]; then
    echo "ERROR: reflection directory does not exist:"
    echo "  $REFL_DIR"
    exit 1
fi

###############################################################################
# Clean environment before cctbx / psana2
###############################################################################

unset PYTHONHOME
unset PYTHONNOUSERSITE
unset LD_LIBRARY_PATH
unset PYTHONPATH
hash -r

###############################################################################
# Activate cctbx / psana2
###############################################################################

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$REPO"

###############################################################################
# Environment information
###############################################################################

echo "============================================================"
echo "MFX SQRT epoch-24 reflection write-back"
echo "Job ID:       $SLURM_JOB_ID"
echo "Host:         $(hostname)"
echo "Started:      $(date --iso-8601=seconds)"
echo "Run:          $RUN_DIR"
echo "Prediction:   $PRED_NPZ"
echo "Metadata:     $DATA_DIR/metadata.npy"
echo "REFL dir:     $REFL_DIR"
echo "Output:       $WB_OUT"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Memory:       128G"
echo "dials.python: $(which dials.python)"
echo "============================================================"

###############################################################################
# Write back predictions
###############################################################################

START=$(date +%s)

/usr/bin/time -v \
dials.python "$REPO/workflows/mfx101555026/scripts/writeback/run_writeback.py" \
    --npz "$PRED_NPZ" \
    --metadata "$DATA_DIR/metadata.npy" \
    --refl-dir "$REFL_DIR" \
    --out-dir "$WB_OUT"

END=$(date +%s)
ELAPSED=$((END - START))

###############################################################################
# Verify output
###############################################################################

if [[ ! -d "$WB_OUT" ]]; then
    echo "ERROR: expected write-back directory was not created:"
    echo "  $WB_OUT"
    exit 1
fi

###############################################################################
# Summary
###############################################################################

echo
echo "============================================================"
echo "Write-back completed successfully."
echo "Finished: $(date --iso-8601=seconds)"

printf "Elapsed: %02d:%02d:%02d\n" \
    $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) \
    $((ELAPSED % 60))

echo
echo "Output directory:"
echo "  $WB_OUT"

echo
echo "Top-level output files and symlinks:"
find "$WB_OUT" -maxdepth 1 \( -type f -o -type l \) | wc -l

echo "============================================================"