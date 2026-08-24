#!/bin/bash
# Validate the config, dataset, and that the model builds. Seconds, no GPU.
# Run this on a login node before submitting anything.
#
# The pre-flight is mode-aware (scripts/preflight.py): it reads mode from the
# config and applies rotation-data rules (one wavelength for the run, DIALS
# scales from the experiment list, so no per-reflection wavelength or image_num
# is required). --pipeline-cfg is omitted here: it only adds poly downstream
# tool checks (careless), which do not apply to the mono DIALS/phenix path.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

python "$INTEGRATOR_ROOT/scripts/preflight.py" \
    --config "$CONFIG"
