#!/usr/bin/env bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=0-8:00
#SBATCH --job-name=mfx_5000_box
#SBATCH --array=0-15
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx_5000_box_%A_%a.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx_5000_box_%A_%a.err

set -eo pipefail

###############################################################################
# Environment
###############################################################################

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

# Avoid "Too many open files" errors from psana2/dxtbx.
ulimit -n 65535
echo "Open file limit: $(ulimit -n)"

export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:${PYTHONPATH:-}

cd /sdf/home/t/thaoh/s3df_practice/integrator

###############################################################################
# Paths
###############################################################################

BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx

DATA_DIR="$BASE/all_runs_269_289_no275_024_rg101_integrated"
OUT_BASE="$BASE/mfx_shoebox_5000_269_289_no275_024_rg101_chunks"
MASK="$BASE/masks/r0269_pam_stddev_border2_combined.mask"

mkdir -p "$OUT_BASE"
mkdir -p "$BASE/logs"

###############################################################################
# Split the first 5,000 source files across 16 Slurm array tasks
###############################################################################

TOTAL_FILES=5000
N_TASKS=16
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Ceiling division:
# 5000 files / 16 tasks gives about 313 files per task.
FILES_PER_TASK=$(( (TOTAL_FILES + N_TASKS - 1) / N_TASKS ))

START=$(( TASK_ID * FILES_PER_TASK ))
END=$(( START + FILES_PER_TASK ))

if [ "$END" -gt "$TOTAL_FILES" ]; then
    END=$TOTAL_FILES
fi

COUNT=$(( END - START ))

OUT_DIR="$OUT_BASE/chunk_${TASK_ID}_start_${START}_n_${COUNT}"

###############################################################################
# Print task information
###############################################################################

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Array task ID: $TASK_ID"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "TOTAL_FILES=$TOTAL_FILES"
echo "N_TASKS=$N_TASKS"
echo "FILES_PER_TASK=$FILES_PER_TASK"
echo "START=$START"
echo "END=$END"
echo "COUNT=$COUNT"
echo "DATA_DIR=$DATA_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "MASK=$MASK"
echo "============================================================"

if [ "$COUNT" -le 0 ]; then
    echo "No files assigned to this task. Exiting."
    exit 0
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: data directory does not exist:"
    echo "$DATA_DIR"
    exit 1
fi

if [ ! -f "$MASK" ]; then
    echo "ERROR: detector mask does not exist:"
    echo "$MASK"
    exit 1
fi

###############################################################################
# Create shoeboxes for this task's file range
###############################################################################

dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --mfx-pattern "*_integrated.refl" \
  --start-file "$START" \
  --max-files "$COUNT" \
  --detector-mask "$MASK" \
  --w 25 \
  --h 25 \
  --d 1 \
  --counts-dtype float32 \
  --write-chunk-size 10000 \
  --no-mask-overlap

echo "Finished task $TASK_ID at $(date)"
echo "Output: $OUT_DIR"