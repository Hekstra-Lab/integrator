#!/bin/bash
# Train the integrator on the shoeboxes this dataset produced.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

RUN_NAME="${RUN_NAME:-sbgrid_${SBGRID_ID}}"
CONFIG="${CONFIG:-$KIT_DIR/config.yaml}"

[[ -f "$CONFIG" ]] || { echo "no config at $CONFIG; write one from the dataset card" >&2; exit 1; }

OUT="$OUT" CONFIG="$CONFIG" RUN_NAME="$RUN_NAME" \
WB_PROJECT="$WB_PROJECT" WB_SAVE_DIR="$WB_SAVE_DIR" \
NUM_WORKERS="${NUM_WORKERS:-6}" EPOCHS="${EPOCHS:-40}" \
    sbatch -p "$GPU_PARTITION" --gres=gpu:1 -t 12:00:00 -c 7 --mem=60G \
    "$INTEGRATOR_ROOT/scripts/poly/cannon/02b_train_requeue.sh"
