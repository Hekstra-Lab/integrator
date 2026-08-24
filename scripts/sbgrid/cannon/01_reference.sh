#!/bin/bash
# DIALS reference processing: import through merge, per sweep then joint.
# CPU only. Submits and returns; watch with squeue.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

DATA_DIR="$DATA_DIR" INTEGRATOR_ROOT="$INTEGRATOR_ROOT" \
    sbatch -p "$CPU_PARTITION" ${MAX_IMAGES:+--export=ALL,MAX_IMAGES=$MAX_IMAGES} \
    "$INTEGRATOR_ROOT/scripts/sbgrid/reference.slurm"
