#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=lcls:prjlumine22
#SBATCH --qos=normal
#SBATCH --constraint=OS_VER:8.6
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --mem=96G
#SBATCH -t 0-12:00
#SBATCH -o /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/gamma_full_sequential_%j.out
#SBATCH -e /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/gamma_full_sequential_%j.err

set -eo pipefail
export PS1="${PS1:-}"

################################################################################
# Paths
################################################################################

REPO=/sdf/home/t/thaoh/s3df_practice/integrator
CONFIG_DIR=$REPO/workflows/mfx101555026/configs

BASE=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx
RUNS=$BASE/runs
LOGS=$BASE/logs
DATASET=$BASE/mfx_shoebox_r0269_018_rg070_fastreader_merged_all

mkdir -p "$RUNS"
mkdir -p "$LOGS"

################################################################################
# Environment setup
################################################################################

cd "$REPO"

export MAMBA_ROOT_PREFIX=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/micromamba_root
eval "$(/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/hewl_1118_tutorial/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate integrator-cuda-dev

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$REPO/src

echo "========================================"
echo "Gamma full sequential training/prediction"
echo "Time: $(date)"
echo "Node: $HOSTNAME"
echo "Repo: $REPO"
echo "Runs folder: $RUNS"
echo "Dataset: $DATASET"
echo "========================================"

echo "Python: $(which python)"
echo "integrator.train: $(which integrator.train)"
echo "integrator.predict: $(which integrator.predict)"

echo "CUDA check:"
python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "nvidia-smi:"
nvidia-smi

################################################################################
# Configs to run sequentially
################################################################################

CONFIGS=(
  #"$CONFIG_DIR/mfx_asinh_normal_gamma.yaml"
  #"$CONFIG_DIR/mfx_softplus_normal_gamma.yaml"
  "$CONFIG_DIR/mfx_sqrtsqureplus_normal_gamma.yaml"
)

NAMES=(
  #"asinh_normal_gamma"
  #"softplus_normal_gamma"
  "sqrt_squareplus_normal_gamma"
)

################################################################################
# Train + predict each config
################################################################################

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"

    echo ""
    echo "################################################################################"
    echo "# Starting $NAME"
    echo "# Config: $CONFIG"
    echo "# Time: $(date)"
    echo "################################################################################"

    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: Config file does not exist:"
        echo "$CONFIG"
        exit 1
    fi

    echo "Checking dataset path mentions in config:"
    grep -n "mfx_shoebox" "$CONFIG" || true
    grep -n "metadata" "$CONFIG" || true

    BEFORE_FILE="$LOGS/${NAME}_before_runs_${SLURM_JOB_ID}.txt"
    AFTER_FILE="$LOGS/${NAME}_after_runs_${SLURM_JOB_ID}.txt"
    RUN_DIR_FILE="$LOGS/${NAME}_run_dir_${SLURM_JOB_ID}.txt"

    echo "Recording existing run folders before training..."
    find "$RUNS" -maxdepth 1 -type d -name "run_*" | sort > "$BEFORE_FILE"

    echo "Training started for $NAME: $(date)"
    integrator.train \
      --config "$CONFIG" \
      --log-dir "$RUNS"

    echo "Training finished for $NAME: $(date)"

    echo "Recording run folders after training..."
    find "$RUNS" -maxdepth 1 -type d -name "run_*" | sort > "$AFTER_FILE"

    echo "Finding new run directory..."
    RUN_DIR=$(comm -13 "$BEFORE_FILE" "$AFTER_FILE" | head -n 1)

    if [[ -z "${RUN_DIR:-}" ]]; then
        echo "WARNING: No new run found by before/after comparison."
        echo "Falling back to newest run_* under $RUNS"
        RUN_DIR=$(find "$RUNS" -maxdepth 1 -type d -name "run_*" -printf "%T@ %p\n" | sort -nr | head -n 1 | awk '{print $2}')
    fi

    if [[ -z "${RUN_DIR:-}" ]]; then
        echo "ERROR: could not find run directory under $RUNS"
        exit 1
    fi

    echo "Using run directory for $NAME:"
    echo "$RUN_DIR"

    echo "$RUN_DIR" > "$RUN_DIR_FILE"

    echo "Prediction started for $NAME: $(date)"
    integrator.predict \
      --run-dir "$RUN_DIR" 

    echo "Prediction finished for $NAME: $(date)"

    echo "Prediction folders for $NAME:"
    find "$RUN_DIR/predictions" -maxdepth 2 -type d || true

    echo "Finished $NAME"
    echo "Run directory saved to:"
    echo "$RUN_DIR_FILE"
done

echo ""
echo "========================================"
echo "All gamma full jobs finished."
echo "Finished time: $(date)"
echo "Run directory pointer files:"
ls -lh "$LOGS"/*_run_dir_${SLURM_JOB_ID}.txt || true
echo "========================================"
