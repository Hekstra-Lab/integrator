#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p sapphire,shared
#SBATCH --mem=64G
#SBATCH -t 0-12:00
#SBATCH -o dials_ref_%j.out
#SBATCH -e dials_ref_%j.err
#
# DIALS reference processing for an SBGrid dataset. CPU only, so it neither
# competes with training nor spends one of the two gpu_test slots.
#
#   DATA_DIR=<dataset dir> sbatch scripts/sbgrid/reference.slurm
#   DATA_DIR=<dir> MAX_IMAGES=50 sbatch scripts/sbgrid/reference.slurm
#
# MAX_IMAGES is a rehearsal: it exercises every step on a few frames per
# sweep in minutes, which is worth doing before spending hours integrating
# 4,320 of them.

set -euo pipefail

INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set.}"
DATA_DIR="${DATA_DIR:?DATA_DIR must be set to the downloaded dataset.}"
MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
# integrator-cuda-dev, not the `dials` env: that one predates the storage
# migration and its console scripts still carry #!/n/hekstra_lab shebangs, so
# exec fails with a FileNotFoundError naming the *script*, which reads as
# "dials.import is missing" when the interpreter is what is missing.
ENV_NAME="${ENV_NAME:-integrator-cuda-dev}"

source "$MAMBA_SH"
micromamba activate "$ENV_NAME"

args=(--data-dir "$DATA_DIR" --nproc "${SLURM_CPUS_PER_TASK:-8}")
[[ -n "${MAX_IMAGES:-}" ]] && args+=(--max-images "$MAX_IMAGES")
[[ -n "${OUT_DIR:-}" ]] && args+=(--out-dir "$OUT_DIR")
[[ -n "${D_MIN:-}" ]] && args+=(--d-min "$D_MIN")

python "$INTEGRATOR_ROOT/scripts/sbgrid/dials_reference.py" "${args[@]}"
