#!/bin/bash
# predict -> careless config4 -> phenix.refine x2 -> anomalous peaks.
#
#   ./03_pipeline.sh <run-dir>                 # submit
#   ./03_pipeline.sh <run-dir> --dry-run       # print the four commands, run nothing
#   STEPS=careless,phenix,peaks ./03_pipeline.sh <run-dir>
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

run_dir="${1:?Usage: ./03_pipeline.sh <run-dir> [--dry-run]}"
shift || true

if [[ "${1:-}" == "--dry-run" ]]; then
    source "$MAMBA_SH"
    micromamba activate "$ENV_DRIVER"
    exec python "$INTEGRATOR_ROOT/scripts/poly/run_pipeline.py" \
        --run-dir "$run_dir" --cfg "$PIPELINE_CFG" --dry-run
fi

# predict and careless both want a GPU. gpu_test is the partition that
# actually schedules here, and it caps a one-GPU job at under 8 cores and
# under ~64 GB, so the same -c 7 / --mem=60G as the training jobs.
RUN_DIR="$run_dir" \
STEPS="${STEPS:-predict,careless,phenix,peaks}" \
PIPELINE_CFG="$PIPELINE_CFG" \
    sbatch -p "${PARTITION:-gpu_test}" --gres=gpu:1 \
        -t "${WALLTIME:-12:00:00}" -c 7 --mem=60G \
        "$INTEGRATOR_ROOT/scripts/poly/pipeline.slurm"
