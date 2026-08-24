#!/bin/bash
# Validate the config, dataset, and every downstream path. Seconds, no GPU.
# Run this on a login node before submitting anything.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

python "$INTEGRATOR_ROOT/scripts/preflight.py" \
    --config "$CONFIG" \
    --pipeline-cfg "$PIPELINE_CFG"
