#!/bin/bash
# Mono downstream: predict -> DIALS scale+merge -> phenix.refine -> rs.find_peaks
# -> merging_stats.csv, for one trained run. This is the mono analog of poly's
# pipeline.slurm; it diverges because mono scales/merges with DIALS and refines
# with phenix (process_single.py), where poly uses careless.
#
# Predict needs the GPU; DIALS/phenix run inline on the same node.
#
# SCORE_EPOCHS selects which checkpoints are scored, through the same
# `integrator.utils.checkpoints` the polychromatic driver uses:
#
#   last       the final checkpoint only
#   every:<n>  every n-th epoch, and always the last
#   all        every checkpoint
#   <int>      that one epoch
#
# The monochromatic chain is DIALS and phenix on CPU, so a trajectory is
# affordable here in a way it is not for careless; `every:5` is the usual
# choice. post_config.py then globs every `preds*.refl` that predict wrote,
# so the loop below picks up each epoch without further argument.
#
#   RUN_DIR=/scratch/.../mono_hewl9b7c ./03_pipeline.sh
#   SCORE_EPOCHS=every:5 ./03_pipeline.sh /scratch/.../mono_hewl9b7c
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

RUN_DIR="${RUN_DIR:-${1:?Usage: RUN_DIR=<run> ./03_pipeline.sh  (or pass it as \$1)}}"
SCORE_EPOCHS="${SCORE_EPOCHS:-last}"

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

mapfile -t ckpts < <(python -m integrator.utils.checkpoints "$RUN_DIR" --scope "$SCORE_EPOCHS")
[[ ${#ckpts[@]} -gt 0 ]] || { echo "ERROR: no checkpoint matched SCORE_EPOCHS=$SCORE_EPOCHS in $RUN_DIR" >&2; exit 1; }

echo "===== predict ($SCORE_EPOCHS -> ${#ckpts[@]} checkpoint(s)) ====="
for ckpt in "${ckpts[@]}"; do
    echo "--- $(basename "$ckpt") ---"
    integrator.predict -v --run-dir "$RUN_DIR" --ckpt "$ckpt" --write-refl
done

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
