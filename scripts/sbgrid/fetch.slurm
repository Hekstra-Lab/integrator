#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --mem=8G
#SBATCH -t 0-08:00
#SBATCH -o sbgrid_fetch_%j.out
#SBATCH -e sbgrid_fetch_%j.err
#
# Download an SBGrid dataset. Runs as a batch job rather than in a login
# shell because a 27 GB transfer outlives an ssh session, and login nodes
# are not for sustained transfers.
#
#   SBGRID_ID=821 sbatch scripts/sbgrid/fetch.slurm
#   SBGRID_ID=821 MAX_FILES=20 sbatch scripts/sbgrid/fetch.slurm   # subset
#
# Resumable: rsync skips what is already there, so resubmitting after a
# timeout costs a listing rather than a re-transfer.
#
# Deliberately on `shared`, not a GPU partition: gpu_test allows only 2
# submitted jobs per user across all sessions, and a download has no reason
# to spend one of them.

set -euo pipefail

INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set.}"
SBGRID_ID="${SBGRID_ID:?SBGRID_ID must be set, e.g. SBGRID_ID=821}"
SBGRID_ROOT="${SBGRID_ROOT:-/n/netscratch/hekstra_lab/Lab/laldama/sbgrid}"
MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
ENV_NAME="${ENV_NAME:-integrator-cuda-dev}"

export SBGRID_ROOT

source "$MAMBA_SH"
micromamba activate "$ENV_NAME"

args=(--id "$SBGRID_ID")
[[ -n "${MAX_FILES:-}" ]] && args+=(--max-files "$MAX_FILES")

python "$INTEGRATOR_ROOT/scripts/sbgrid/fetch.py" "${args[@]}"

echo "===== fetched to $SBGRID_ROOT/$SBGRID_ID"
