#!/bin/bash
#SBATCH --job-name=mfx_fastreader_270
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-2:00
#SBATCH --array=0-15
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx_fastreader_r0270_019_%A_%a.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx_fastreader_r0270_019_%A_%a.err

set -eo pipefail
export PS1="${PS1:-}"

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
ulimit -n 65535

export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH
cd /sdf/home/t/thaoh/s3df_practice/integrator

BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
IN=$BASE/outputs/r0270/019_rg070/out
OUT_BASE=$BASE/mfx_shoebox_r0270_019_rg070_fastreader_chunks
LOGDIR=$BASE/logs

mkdir -p "$OUT_BASE" "$LOGDIR"

CHUNK_SIZE=126
START_FILE=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
OUT=$OUT_BASE/chunk_${SLURM_ARRAY_TASK_ID}

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "START_FILE=$START_FILE"
echo "CHUNK_SIZE=$CHUNK_SIZE"
echo "OUT=$OUT"
echo "ulimit -n=$(ulimit -n)"
date

time dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir "$IN" \
  --out-dir "$OUT" \
  --mfx-pattern "idx-data_*_integrated.refl" \
  --start-file "$START_FILE" \
  --max-files "$CHUNK_SIZE" \
  --detector-mask /sdf/data/lcls/ds/mfx/mfx101555026/results/pam/hot_lines_combined5.mask \
  --w 25 \
  --h 25 \
  --d 1 \
  --counts-dtype float32 \
  --write-chunk-size 50000 \
  --no-stats

date
echo "done"