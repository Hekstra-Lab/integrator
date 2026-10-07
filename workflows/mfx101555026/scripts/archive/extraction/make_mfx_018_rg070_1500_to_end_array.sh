#!/usr/bin/env bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=0-4:00
#SBATCH --job-name=mfx018box2
#SBATCH --array=0-7
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx018_box2_%A_%a.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx018_box2_%A_%a.err

set -eo pipefail

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

# Important: avoid "Too many open files" error from psana2/dxtbx
ulimit -n 8192
echo "open file limit: $(ulimit -n)"

export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH

cd /sdf/home/t/thaoh/s3df_practice/integrator

BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
DATA_DIR=$BASE/outputs/r0269/018_rg070/out
OUT_BASE=$BASE/mfx_shoebox_r0269_018_rg070_chunks_1500_to_end
MASK=/sdf/data/lcls/ds/mfx/mfx101555026/results/pam/hot_lines_combined5.mask

mkdir -p "$OUT_BASE"
mkdir -p "$BASE/logs"

# Files 0-1499 were already processed.
# These 8 jobs process files 1500-4068.
STARTS=(1500 1821 2142 2463 2784 3105 3426 3747)
COUNTS=(321 321 321 321 321 321 321 322)

START=${STARTS[$SLURM_ARRAY_TASK_ID]}
COUNT=${COUNTS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=$OUT_BASE/chunk_${SLURM_ARRAY_TASK_ID}_start_${START}_n_${COUNT}

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "START=$START"
echo "COUNT=$COUNT"
echo "OUT_DIR=$OUT_DIR"
echo "DATA_DIR=$DATA_DIR"

dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --mfx-pattern "idx-data_*_integrated.refl" \
  --start-file "$START" \
  --max-files "$COUNT" \
  --detector-mask "$MASK" \
  --w 25 \
  --h 25 \
  --d 1 \
  --counts-dtype float32 \
  --write-chunk-size 10000 \
  --no-mask-overlap

echo "Done: $OUT_DIR"
