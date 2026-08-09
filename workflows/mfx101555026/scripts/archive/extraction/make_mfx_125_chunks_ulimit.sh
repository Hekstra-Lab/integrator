#!/bin/bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --job-name=mfx125u
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-6:00
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx125u_%A_%a.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/mfx125u_%A_%a.err

ulimit -n 4096
echo "ulimit -n: $(ulimit -n)"

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH
cd /sdf/home/t/thaoh/s3df_practice/integrator

CHUNK=$(printf "%03d" $SLURM_ARRAY_TASK_ID)
IN_DIR=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/chunk_inputs/chunk_${CHUNK}
OUT_DIR=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_chunks/chunk_${CHUNK}

dials.python src/integrator/cli/make_shoeboxes.py \
  --mfx \
  --data-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --mfx-pattern "idx-data_*_integrated.refl" \
  --max-files 125 \
  --detector-mask /sdf/data/lcls/ds/mfx/mfx101555026/results/pam/hot_lines_combined5.mask \
  --w 25 \
  --h 25 \
  --d 1 \
  --counts-dtype float32 \
  --no-mask-overlap
