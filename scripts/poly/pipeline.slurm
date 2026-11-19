#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --mem=64G
#SBATCH -t 0-08:00
#SBATCH -o poly_pipeline_%j.out
#SBATCH -e poly_pipeline_%j.err
#
# Post-training only: predict -> careless -> phenix.refine -> rs.find_peaks.
# The predict step wants a GPU; submit to a GPU partition when including it.
#
# SUBMIT:
#   RUN_DIR=<run-dir> sbatch scripts/poly/pipeline.slurm
#   RUN_DIR=<run-dir> STEPS=careless,phenix,peaks sbatch scripts/poly/pipeline.slurm

set -euo pipefail

INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set (add to ~/.bashrc).}"
RUN_DIR="${RUN_DIR:?RUN_DIR must be set to the trained run directory.}"
PIPELINE_CFG="${PIPELINE_CFG:-$INTEGRATOR_ROOT/scripts/poly/poly_pipeline_cfg.yaml}"
STEPS="${STEPS:-predict,careless,phenix,peaks}"
MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
export MAMBA_SH

# `integrator-cuda-dev`, not `integrator`, and the interpreter by absolute
# path: the latter's console scripts carry a pre-migration shebang that no
# longer exists, so anything resolved through one dies with "bad
# interpreter". The same fix as `train.slurm`.
ENV_DRIVER="${ENV_DRIVER:-integrator-cuda-dev}"
MAMBA_ROOT="${MAMBA_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/micromamba}"
source "$MAMBA_SH"
micromamba activate "$ENV_DRIVER"
PY_BIN="$MAMBA_ROOT/envs/$ENV_DRIVER/bin/python"
[ -x "$PY_BIN" ] || { echo "no interpreter at $PY_BIN"; exit 1; }

# The imported package and this checkout are two different git trees here;
# a mismatch is how six architecture arms died on `Unknown integrator name`.
SRC_DIR=$("$PY_BIN" -c 'import integrator, pathlib; print(pathlib.Path(integrator.__file__).parents[2])' 2>/dev/null)
PKG_SHA=$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
REPO_SHA=$(git -C "$INTEGRATOR_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "===== python:  $PY_BIN"
echo "===== source:  $SRC_DIR ($PKG_SHA)"
if [ "$PKG_SHA" != "$REPO_SHA" ]; then
  echo "===== WARNING: checkout is $REPO_SHA, imported package is $PKG_SHA"
fi

"$PY_BIN" "$INTEGRATOR_ROOT/scripts/poly/run_pipeline.py" \
    --run-dir "$RUN_DIR" \
    --cfg "$PIPELINE_CFG" \
    --steps "$STEPS"
