#!/bin/bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00
#SBATCH --job-name=preprocess_sqrt
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/preprocess_sqrt_%j.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/preprocess_sqrt_%j.err

set -eo pipefail

REPO=/sdf/home/t/thaoh/s3df_practice/integrator
DATA_DIR=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/mfx_shoebox_allruns_269_289_no275_024_rg101
SQRT_CHUNK_DIR="$DATA_DIR/chunks_sqrt_squareplus"

mkdir -p /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs

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

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Data directory: $DATA_DIR"
echo "Output directory: $SQRT_CHUNK_DIR"
echo "Python: $(which python)"
echo "integrator.preprocess: $(which integrator.preprocess)"

if [[ -e "$SQRT_CHUNK_DIR" ]]; then
    echo "ERROR: output directory already exists:"
    echo "  $SQRT_CHUNK_DIR"
    echo "Remove it or choose a new directory before rerunning."
    exit 1
fi

integrator.preprocess \
  --data-dir "$DATA_DIR" \
  --validation-split 0.2 \
  --split-seed 42 \
  --transform sqrt_squareplus \
  --chunk-size 50000 \
  --n-bins 1 \
  --out-dir "$SQRT_CHUNK_DIR" \
  --min-valid-pixels 10 \
  -v

echo "Finished: $(date)"