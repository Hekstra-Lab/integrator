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
#   SRC=/path/to/checkout/src sbatch scripts/poly/train.slurm   # pin the source tree
#
# The log records the interpreter, the source tree, its commit, the node and
# the CPU count, because none of those are implied by the config and all
# four change the result or the time it takes.

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
# `integrator-cuda-dev`, not `integrator`: the latter's console scripts carry
# a pre-migration shebang (`/n/hekstra_lab/people/aldama/...`) that no longer
# exists, so `integrator.train` there dies with "bad interpreter" before any
# of this script's arguments are read. That is why the jobs this script is
# the front door for were being launched from one-off sbatch files instead.
ENV_TRAIN="${ENV_TRAIN:-integrator-cuda-dev}"
MAMBA_ROOT="${MAMBA_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/micromamba}"
# A specific source tree to run, ahead of whatever the environment installed.
# Four checkouts answer to `import integrator` on this cluster; a run that
# does not record which one it used cannot be reproduced.
SRC="${SRC:-}"

run_dir="$OUT/$RUN_NAME"
mkdir -p "$run_dir"

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

# The environment's own interpreter, by absolute path. Entry points resolve
# through a shebang written at install time, which outlived the storage
# migration; the interpreter does not.
PY="$MAMBA_ROOT/envs/$ENV_TRAIN/bin/python"
[[ -x "$PY" ]] || { echo "no interpreter at $PY"; exit 1; }
[[ -n "$SRC" ]] && export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

echo "===== config:  $CONFIG"
echo "===== run dir: $run_dir"
echo "===== python:  $PY"
# Which source tree actually ran, and which commit, into the log. Reading a
# result later means knowing this.
"$PY" -c 'import integrator, pathlib; print("===== source: ", pathlib.Path(integrator.__file__).parent)'
( cd "$(dirname "$("$PY" -c 'import integrator;print(integrator.__file__)')")/../.." \
  && echo "===== commit:  $(git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
                      "$(git diff --quiet 2>/dev/null || echo '(dirty)')" ) || true
echo "===== node:    $(hostname), -c ${SLURM_CPUS_PER_TASK:-?}"

# Fail before the allocation is spent, not four hours in.
"$PY" "$INTEGRATOR_ROOT/scripts/preflight.py" --config "$CONFIG"

args=(--config "$CONFIG" --run-dir "$run_dir" -v)
[[ -n "$EPOCHS" ]]     && args+=(--max-epochs "$EPOCHS")
[[ -n "$WB_PROJECT" ]] && args+=(--wb-project "$WB_PROJECT")
[[ "$FIGURES" == "1" ]] && args+=(--figures)

"$PY" -m integrator.cli.train "${args[@]}"

if [[ "$RUN_PIPELINE" == "1" ]]; then
    echo "===== post-training pipeline"
    "$PY" "$INTEGRATOR_ROOT/scripts/poly/run_pipeline.py" \
        --run-dir "$run_dir" \
        --cfg "${PIPELINE_CFG:-$INTEGRATOR_ROOT/scripts/poly/poly_pipeline_cfg.yaml}"
fi

echo "===== done: $run_dir"
