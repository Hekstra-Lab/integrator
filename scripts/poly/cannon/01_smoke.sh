#!/bin/bash
# 2 epochs on 50k reflections: proves the whole chain works before the long run.
# Takes minutes. Follow it with 03_pipeline.sh using the smoke pipeline config.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

CONFIG="$OUT/hewl1118_smoke.yaml" \
RUN_NAME="smoke" \
EPOCHS=2 \
FIGURES=1 \
    sbatch --time=0-02:00 "$INTEGRATOR_ROOT/scripts/poly/train.slurm"
