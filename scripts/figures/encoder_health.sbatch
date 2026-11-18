#!/bin/bash
#SBATCH -J enc_health
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 0-02:00
#SBATCH -o enc_health_%j.out
#
# Measure whether the profile encoder is alive, over training.
#
# This is a batch job rather than an ssh one-liner because it needs the
# DATASET, not just the checkpoints. `dump_basis.py` reads state dicts and
# runs fine on a login node; this one runs the model's forward on real
# shoeboxes, and the data module loads counts.npy whole -- about 1.75 GB
# for PDZ2's 993,143 reflections, plus masks, plus torch. Run on the login
# node it was killed during loading, before printing a line, which looks
# exactly like a hang.
#
#   DATASET=pdz2_1116 sbatch scripts/figures/encoder_health.sbatch \
#       /path/to/runs/pdz2_svae /path/to/runs/pdz2_w21_k7m
#
# STRIDE=n   every nth checkpoint (default every one; 25 is a quick look)
# OUT=path   where the CSV goes (default enc_<dataset>.csv beside the log)
set -uo pipefail
DATASET=${DATASET:?set DATASET to the dataset these runs trained on}
STRIDE=${STRIDE:-1}
OUT=${OUT:-$PWD/enc_${DATASET}.csv}
ENV_NAME=${ENV_NAME:-integrator-cuda-dev}
MAMBA_ROOT=${MAMBA_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/micromamba}
PY=$MAMBA_ROOT/envs/$ENV_NAME/bin/python
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 1; }

# The script is taken from the checkout this job was submitted from, and run
# from a directory with no `integrator/` in it: Python puts the script's own
# directory first on sys.path, and a staged stub package under `jobs/` has
# shadowed the installed one twice.
HERE=${SLURM_SUBMIT_DIR:?submit from a checkout}
SCRIPT=$HERE/scripts/figures/dump_encoder_health.py
[ -s "$SCRIPT" ] || { echo "no $SCRIPT"; exit 1; }

echo "===== python:  $PY"
echo "===== dataset: $DATASET, stride $STRIDE"
echo "===== runs:    $*"
"$PY" -c 'import integrator, pathlib; print("===== source: ", pathlib.Path(integrator.__file__).parent)'
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

cd /tmp || exit 1
DATASET=$DATASET STRIDE=$STRIDE "$PY" -u "$SCRIPT" "$OUT" "$@"
echo "ENC_HEALTH_DONE"
