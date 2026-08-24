#!/usr/bin/env bash
# Process a SINGLE integrator checkpoint end-to-end:
#   1. predict + write a .refl file for one epoch's checkpoint
#   2. run the DIALS/Phenix post-processing on that checkpoint's predictions
#
# `run_dir` must contain run_paths.yaml (written by integrator.train); the
# checkpoint for --epoch is located through that metadata's wandb log_dir, and
# predictions are written under <log_dir>/../predictions/epoch_XXXX.
#
# To extend to ALL checkpoints later, drop --epoch: integrator.pred then loops
# over every epoch*.ckpt in the run.
#
# Usage:
#   scripts/process_checkpoint.sh --run-dir DIR --epoch N \
#       --refltorch-dir /path/to/refltorch [options]
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: process_checkpoint.sh --run-dir DIR --epoch N [options]

Required:
  --run-dir DIR          run directory containing run_paths.yaml
  --epoch N              checkpoint epoch to process (single-checkpoint mode)

DIALS/Phenix step:
  --refltorch-dir DIR    refltorch repo with create_config.py + submit_jobs.py
                         (or set REFLTORCH_DIR)

Environments (micromamba):
  --integrator-env NAME  env providing `integrator.pred`  (default: integrator)
  --refltorch-env NAME   env for DIALS/Phenix             (default: refltorch)

DIALS/Phenix tool paths (forwarded to create_config.py; else its env vars
DIALS_ENV / PHENIX_ENV are used):
  --dials-env PATH       DIALS activate script
  --phenix-env PATH      Phenix env script

Flags:
  --skip-predict         skip step 1 (the .refl already exists)
  --skip-phenix          skip step 2 (prediction only)
  -h, --help             show this help

Notes:
  * create_config.py globs ALL predictions/**/preds*.refl, so single-checkpoint
    isolation requires that the run has no other epochs' predictions yet
    (use a fresh run_dir or clean the predictions/ dir first).
  * submit_jobs.py submits a SLURM array (one task per .refl) + a dependent
    plotting job, so step 2 only runs on the cluster (needs sbatch).
  * --refltorch-dir is the dials_output scripts dir (create_config.py,
    submit_jobs.py, process_single_refl.py, compare_models.py).
EOF
}

run_dir=""
epoch=""
refltorch_dir="${REFLTORCH_DIR:-}"
integrator_env="${INTEGRATOR_ENV:-integrator}"
refltorch_env="${REFLTORCH_ENV:-refltorch}"
dials_env=""
phenix_env=""
skip_predict=0
skip_phenix=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)        run_dir="$2"; shift 2;;
    --epoch)          epoch="$2"; shift 2;;
    --refltorch-dir)  refltorch_dir="$2"; shift 2;;
    --integrator-env) integrator_env="$2"; shift 2;;
    --refltorch-env)  refltorch_env="$2"; shift 2;;
    --dials-env)      dials_env="$2"; shift 2;;
    --phenix-env)     phenix_env="$2"; shift 2;;
    --skip-predict)   skip_predict=1; shift;;
    --skip-phenix)    skip_phenix=1; shift;;
    -h|--help)        usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1;;
  esac
done

# --- validation -------------------------------------------------------------
[[ -n "$run_dir" ]] || { echo "ERROR: --run-dir is required" >&2; usage; exit 1; }
[[ -f "$run_dir/run_paths.yaml" ]] || {
  echo "ERROR: $run_dir/run_paths.yaml not found" >&2; exit 1; }
if [[ "$skip_predict" -eq 0 && -z "$epoch" ]]; then
  echo "ERROR: --epoch is required for single-checkpoint mode" >&2; exit 1
fi

# micromamba shell integration (harmless no-op if already configured)
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)" || true
fi

# --- 1. Predict + write .refl for the single checkpoint ---------------------
if [[ "$skip_predict" -eq 0 ]]; then
  echo "===== Starting integrator.pred (epoch ${epoch}) ====="
  micromamba activate "$integrator_env"
  integrator.pred -v \
      --run-dir "$run_dir" \
      --epoch "$epoch" \
      --write-refl
fi

# --- 2. DIALS/Phenix post-processing ----------------------------------------
# NOTE: create_config.py / submit_jobs.py discover predictions from the run's
# metadata. Predicting a single epoch limits them to that one checkpoint as long
# as the run has no other epoch_* predictions yet. If those scripts accept an
# --epoch / --ckpt selector, pass it through here for stricter isolation.
if [[ "$skip_phenix" -eq 0 ]]; then
  [[ -n "$refltorch_dir" ]] || {
    echo "ERROR: --refltorch-dir (or REFLTORCH_DIR) required for the phenix step" >&2
    exit 1; }

  echo "===== Starting DIALS-Phenix Processing (epoch ${epoch}) ====="
  micromamba deactivate || true
  micromamba activate "$refltorch_env"

  log_dir="${run_dir}/dials_phenix_logs"
  mkdir -p "$log_dir"

  create_config_args=(--run-dir "$run_dir")
  [[ -n "$dials_env" ]]  && create_config_args+=(--dials-env "$dials_env")
  [[ -n "$phenix_env" ]] && create_config_args+=(--phenix-env "$phenix_env")
  python "$refltorch_dir/create_config.py" "${create_config_args[@]}"

  python "$refltorch_dir/submit_jobs.py" \
      --run-dir "$run_dir" \
      --log-dir "$log_dir" \
      --script-dir "$refltorch_dir"
fi

echo "===== Done (epoch ${epoch}) ====="
