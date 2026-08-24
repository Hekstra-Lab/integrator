#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-15:00
#SBATCH -o sfx_%A_%a.out
#SBATCH -e sfx_%A_%a.err
#SBATCH --array=0-5
#
# SFX per-image-Wilson experiment matrix as a SLURM job array: one array task per arm,
# each an independent GPU job writing to $OUT/<tag>/. Resumable -- resubmit the same
# array and every task picks up from its latest checkpoint.
#
# PREREQ (run once, cheap, CPU-only -- do NOT waste a GPU job on it):
#   micromamba activate integrator-cuda-dev
#   cd $REPO
#   python scripts/jungfrau_sim/sfx_generate.py --n-images 2000 --n-refl 100 --out data/sfx_sim
#   # for the per-image-B arm, also: ... --sigma-b 5 --out data/sfx_sim_Bi
#
# SUBMIT:  sbatch scripts/jungfrau_sim/sfx_train.slurm
#   subset:        sbatch --array=0,3 scripts/jungfrau_sim/sfx_train.slurm
#   more epochs:   EPOCHS=800 sbatch scripts/jungfrau_sim/sfx_train.slurm

set -euo pipefail
export TQDM_DISABLE=1

# --- paths / knobs (override via env: DATA=... OUT=... EPOCHS=... sbatch ...) ---
REPO=${REPO:-/n/lab_storage/hekstra_lab/people/aldama/software/integrator}
DATA=${DATA:-$REPO/data/sfx_sim}
OUT=${OUT:-$REPO/data/sfx_runs}
EPOCHS=${EPOCHS:-400}

# --- experiment matrix: one entry per array task (index = SLURM_ARRAY_TASK_ID) ---
ARMS=(
  "--likelihood poisson        --profile learned --scale per_image"   # 0 headline: int+Poisson
  "--likelihood normal_coupled --profile learned --scale per_image"   # 1 headline: real+Normal(coupled)
  "--likelihood normal_free    --profile learned --scale per_image"   # 2 headline: real+Normal(free)
  "--likelihood poisson        --profile known   --scale per_image"   # 3 oracle-profile control
  "--likelihood poisson        --profile learned --scale global"      # 4 per-image G vs global G
  "--likelihood normal_coupled --profile known   --scale per_image"   # 5 oracle control, Normal
  # per-image B needs the --sigma-b dataset; add to the array and point DATA at it:
  # "--likelihood poisson      --profile learned --scale per_image --per-image-B"
)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"

# --- environment ---
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
micromamba activate integrator-cuda-dev

cd "$REPO"
echo "host $(hostname)  gpu $CUDA_VISIBLE_DEVICES  task $SLURM_ARRAY_TASK_ID  arm: $ARM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u scripts/jungfrau_sim/sfx_experiment.py $ARM \
    --data "$DATA" \
    --out "$OUT" \
    --epochs "$EPOCHS" \
    --device cuda \
    --eval-every 5 \
    --resume

echo "done task $SLURM_ARRAY_TASK_ID"
