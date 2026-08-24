#!/bin/bash
# Mono downstream: predict -> DIALS scale+merge -> phenix.refine -> rs.find_peaks
# -> merging_stats.csv, for one trained run. This is the mono analog of poly's
# pipeline.slurm; it diverges because mono scales/merges with DIALS and refines
# with phenix (process_single.py), where poly uses careless.
#
# Single job, no sub-array: predict writes exactly one refl file (the final
# checkpoint), so there is exactly one downstream task. Predict needs the GPU;
# DIALS/phenix run inline on the same node.
#
#   RUN_DIR=/scratch/.../mono_hewl9b7c ./03_pipeline.sh
#   ./03_pipeline.sh /scratch/.../mono_hewl9b7c
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

RUN_DIR="${RUN_DIR:-${1:?Usage: RUN_DIR=<run> ./03_pipeline.sh  (or pass it as \$1)}}"

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

# Resolve the checkpoint dir from run_paths.yaml (with W&B the output root moves
# under the wandb run dir, so <run_dir>/files is only right for local logging).
log_dir=""
if [[ -f "$RUN_DIR/run_paths.yaml" ]]; then
    log_dir=$(python -c 'import sys,yaml; print((yaml.safe_load(open(sys.argv[1])) or {}).get("log_dir",""))' "$RUN_DIR/run_paths.yaml")
fi
ckpt="${log_dir:-$RUN_DIR/files}/checkpoints/last.ckpt"
[[ -e "$ckpt" ]] || { echo "ERROR: no checkpoint at $ckpt" >&2; exit 1; }

echo "===== predict (single checkpoint: $ckpt) ====="
integrator.predict -v --run-dir "$RUN_DIR" --ckpt "$ckpt" --write-refl

echo "===== post_config: write dials_phenix_cfg.yaml ====="
python "$INTEGRATOR_ROOT/scripts/mono/post_config.py" \
    --run-dir "$RUN_DIR" --process-cfg "$PROCESS_CFG"

cfg="$RUN_DIR/dials_phenix_cfg.yaml"
[[ -f "$cfg" ]] || { echo "ERROR: post_config did not write $cfg" >&2; exit 1; }
n=$(python -c 'import sys,yaml; print(len(yaml.safe_load(open(sys.argv[1]))["refl_files"]))' "$cfg")
echo "===== DIALS/phenix/peaks over $n refl file(s) ====="
for ((i = 0; i < n; i++)); do
    echo "--- process_single index $i ---"
    python "$INTEGRATOR_ROOT/scripts/mono/process_single.py" --config "$cfg" --index "$i"
done

echo "===== emit merging_stats.csv ====="
python "$INTEGRATOR_ROOT/scripts/mono/emit_merging_stats.py" --run-dir "$RUN_DIR"

echo "===== done: $RUN_DIR ====="
