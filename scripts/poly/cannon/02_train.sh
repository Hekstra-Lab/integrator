#!/bin/bash
# The real run: full dataset, 100 epochs, figure dumps on.
#
#   ./02_train.sh              # defaults
#   EPOCHS=40 ./02_train.sh    # shorter
#   WB_PROJECT=laue ./02_train.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

RUN_NAME="${RUN_NAME:-hewl1118_$(date +%Y%m%d)}" \
    sbatch "$INTEGRATOR_ROOT/scripts/poly/train.slurm"
