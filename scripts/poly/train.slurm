#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-12:00
#SBATCH -o poly_train_%j.out
#SBATCH -e poly_train_%j.err
#
# Train the Laue integrator, with the training-figure dumps enabled.
#
# PREREQ: export INTEGRATOR_ROOT=<repo> in your shell rc.
# SUBMIT:
#   sbatch scripts/poly/train.slurm
#   CONFIG=configs/poly/hewl1118_poly.yaml EPOCHS=100 sbatch scripts/poly/train.slurm
#   WB_PROJECT=poly_laue sbatch scripts/poly/train.slurm
#   RUN_PIPELINE=1 sbatch scripts/poly/train.slurm   # chain predict->careless->phenix->peaks

set -euo pipefail
export TQDM_DISABLE=1

INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set (add to ~/.bashrc).}"
CONFIG="${CONFIG:-$INTEGRATOR_ROOT/configs/poly/hewl1118_poly.yaml}"
OUT="${OUT:-$PWD}"
RUN_NAME="${RUN_NAME:-poly_${SLURM_JOB_ID:-manual}}"
EPOCHS="${EPOCHS:-}"
WB_PROJECT="${WB_PROJECT:-}"
FIGURES="${FIGURES:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-0}"
MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
ENV_TRAIN="${ENV_TRAIN:-integrator}"

run_dir="$OUT/$RUN_NAME"
mkdir -p "$run_dir"

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

echo "===== config:  $CONFIG"
echo "===== run dir: $run_dir"

# Fail before the allocation is spent, not four hours in.
python "$INTEGRATOR_ROOT/scripts/poly/preflight.py" --config "$CONFIG"

args=(--config "$CONFIG" --run-dir "$run_dir" -v)
[[ -n "$EPOCHS" ]]     && args+=(--max-epochs "$EPOCHS")
[[ -n "$WB_PROJECT" ]] && args+=(--wb-project "$WB_PROJECT")
[[ "$FIGURES" == "1" ]] && args+=(--figures)

integrator.train "${args[@]}"

if [[ "$RUN_PIPELINE" == "1" ]]; then
    echo "===== post-training pipeline"
    python "$INTEGRATOR_ROOT/scripts/poly/run_pipeline.py" \
        --run-dir "$run_dir" \
        --cfg "${PIPELINE_CFG:-$INTEGRATOR_ROOT/scripts/poly/poly_pipeline_cfg.yaml}"
fi

echo "===== done: $run_dir"
