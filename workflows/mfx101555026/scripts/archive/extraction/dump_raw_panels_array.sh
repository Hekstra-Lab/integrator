#!/bin/bash
#SBATCH --job-name=dump_raw_panels
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0-6:00
#SBATCH --array=0-15
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/dump_raw_panels_%A_%a.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/dump_raw_panels_%A_%a.err

set -eo pipefail

export PS1="${PS1:-}"

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

ulimit -n 65535

export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH

cd /sdf/home/t/thaoh/s3df_practice/integrator

BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
IN=$BASE/outputs/r0269/018_rg070/out
CACHE=$BASE/raw_panel_cache_r0269_018_rg070
LOGDIR=$BASE/logs

mkdir -p "$CACHE"
mkdir -p "$LOGDIR"

CHUNK_SIZE=283
START_FILE=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "START_FILE=$START_FILE"
echo "CHUNK_SIZE=$CHUNK_SIZE"
echo "CACHE=$CACHE"
echo "ulimit -n=$(ulimit -n)"
date

dials.python workflows/mfx101555026/scripts/diagnostics/dump_raw_panels_test.py \
  --in-dir "$IN" \
  --out-dir "$CACHE" \
  --start-file "$START_FILE" \
  --max-files "$CHUNK_SIZE"

date
echo "done"
