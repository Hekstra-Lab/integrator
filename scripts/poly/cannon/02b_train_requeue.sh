#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-12:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o poly_train_%j.out
#SBATCH -e poly_train_%j.err
#
# The batch script every front door submits; run_all.sh and
# 02c_train_gputest.sh override the #SBATCH lines below.
#
# PARTITION CHOICE. Prefer gpu_test for a run that fits in 12 hours: it does
# not preempt, and our training runs are ~2h. Its cost is a cap of 2
# submitted jobs per user, shared across sessions.
#
# gpu_requeue, the partition named in the #SBATCH line below, has no submit
# cap and a 3-day limit, but PreemptMode=REQUEUE: a higher-priority job can
# evict yours, which restarts this script on a different GPU. Use it when a
# run needs more than 12 hours, or when both gpu_test slots are taken.
#
# Preemption is survivable but not free. Lightning installs SLURM signal
# handlers and checkpoints when the eviction signal arrives, and this script
# resumes from last.ckpt on restart with the same job id, so the run dir is
# stable. Each eviction still costs a data reload (~3 min for 8.8 GB) and
# whatever training happened since the last checkpoint. Frequent preemption
# can therefore cost more throughput than waiting for a gpu_test slot.
#
# The resume path has not yet been exercised by an actual preemption -- the
# logic is there, but treat the first preempted run as a test of it.
#
#   sbatch 02b_train_requeue.sh
#   EPOCHS=40 sbatch 02b_train_requeue.sh
#
# Submit this file directly with sbatch — it is the batch script, not a
# wrapper, because a requeued job re-runs the script from the top.

set -euo pipefail
export TQDM_DISABLE=1

# SLURM sets SLURM_SUBMIT_DIR to where sbatch ran and preserves it on requeue
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
source ./env.sh

run_dir="$OUT/${RUN_NAME:-hewl1118_requeue_${SLURM_JOB_ID}}"
mkdir -p "$run_dir"

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

echo "===== config:  $CONFIG"
echo "===== run dir: $run_dir"

args=(--config "$CONFIG" --run-dir "$run_dir" -v --figures)
[[ -n "${EPOCHS:-}" ]]     && args+=(--max-epochs "$EPOCHS")
[[ -n "${WB_PROJECT:-}" ]] && args+=(--wb-project "$WB_PROJECT" --save-dir "$WB_SAVE_DIR")
# gpu_test caps cores per GPU, so the worker count has to follow the request
[[ -n "${NUM_WORKERS:-}" ]] && args+=(--num-workers "$NUM_WORKERS")

# A requeued job restarts this script, so pick training back up where it
# stopped. Read the checkpoint directory out of run_paths.yaml rather than
# assuming it: with W&B enabled the output root moves under the wandb run
# dir, so <run_dir>/files is only correct for local-logging runs.
log_dir=""
wb_id=""
if [[ -f "$run_dir/run_paths.yaml" ]]; then
    read -r log_dir wb_id < <(python -c '
import sys, yaml
meta = yaml.safe_load(open(sys.argv[1])) or {}
print(meta.get("log_dir", ""), (meta.get("wandb") or {}).get("run_id", ""))
' "$run_dir/run_paths.yaml")
fi

ckpt="${log_dir:-$run_dir/files}/checkpoints/last.ckpt"
if [[ -e "$ckpt" ]]; then
    echo "===== resuming from $ckpt"
    args+=(--ckpt-path "$ckpt")
fi

# keep every attempt in one W&B run instead of starting a new one per requeue
if [[ -n "${WB_PROJECT:-}" && -n "$wb_id" ]]; then
    echo "===== resuming W&B run $wb_id"
    args+=(--wandb-resume-id "$wb_id")
fi

integrator.train "${args[@]}"

echo "===== done: $run_dir"
