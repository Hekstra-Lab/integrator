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

source "$MAMBA_SH"
micromamba activate "${ENV_DRIVER:-integrator}"

python "$INTEGRATOR_ROOT/scripts/poly/run_pipeline.py" \
    --run-dir "$RUN_DIR" \
    --cfg "$PIPELINE_CFG" \
    --steps "$STEPS"
