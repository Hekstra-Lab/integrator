#!/bin/bash
# Submit the same run to gpu_test — 12 nodes x 8 A100 3g.20gb MIG slices,
# 96 slices total, 12h cap. Usually the shortest wait on this cluster.
#
# gpu_test enforces per-GPU shares of a node (64 cores and 515 GB across 8
# MIG slices): under 8 cores and under 65636 MB per GPU. Hence -c 7 and
# --mem=60G, with 6 dataloader workers to match the core count. The run
# needs ~15 GB resident (counts int32 + its float32 copy + masks), so 60G
# is comfortable; the same shape already loaded fine in the interactive
# session at 60G.
# Everything else, including the resume-from-last.ckpt logic, is shared with
# 02b_train_requeue.sh via command-line overrides of its #SBATCH lines.
#
#   ./02c_train_gputest.sh              # 40 epochs, ~5h on a MIG slice
#   EPOCHS=100 ./02c_train_gputest.sh   # will not finish in 12h; resubmit to continue
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# A fixed run name is what makes the resume chain work: each resubmission
# finds last.ckpt in the same run dir and continues from it.
RUN_NAME="${RUN_NAME:-hewl1118}" \
EPOCHS="${EPOCHS:-40}" \
NUM_WORKERS="${NUM_WORKERS:-6}" \
    sbatch -p gpu_test -t 12:00:00 --mem=60G -c 7 ./02b_train_requeue.sh
