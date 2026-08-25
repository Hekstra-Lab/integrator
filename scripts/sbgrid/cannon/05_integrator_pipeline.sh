#!/bin/bash
# Put the integrator's predictions through the same steps as the reference:
# split per sweep, scale, merge, refine, find peaks.
#
# Every step matches 01_reference.sh and 02_refine.sh, because the comparison
# is only meaningful if the two arms differ in integration and nothing else --
# same scaling program, same model, same R-free set, same peak search.
#
#   ./05_integrator_pipeline.sh <run-dir> [epoch]
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
source "$MAMBA_SH"
micromamba activate "$ENV_MAIN"

run_dir="${1:?Usage: ./05_integrator_pipeline.sh <run-dir> [epoch]}"
epoch="${2:-}"

pred_dir=$(python - "$run_dir" <<'PY'
import sys, yaml, pathlib
meta = yaml.safe_load((pathlib.Path(sys.argv[1]) / "run_paths.yaml").read_text())
print(meta["predictions_dir"])
PY
)
if [[ -n "$epoch" ]]; then
    epoch_dir="$pred_dir/$(printf 'epoch_%04d' "$epoch")"
else
    epoch_dir=$(ls -d "$pred_dir"/epoch_* 2>/dev/null | sort | tail -1)
fi
[[ -d "$epoch_dir" ]] || { echo "no predictions under $pred_dir" >&2; exit 1; }
echo "===== predictions: $epoch_dir"

# one reflection table per sweep, so each is scaled against its own crystal
python "$INTEGRATOR_ROOT/scripts/sbgrid/write_predictions.py" \
    --pred-dir "$epoch_dir" --dataset-dir "$OUT/dataset"

work="$epoch_dir/scaled"
mkdir -p "$work"
mapfile -t refls < <(ls "$epoch_dir"/per_sweep/*.refl)
[[ ${#refls[@]} -gt 0 ]] || { echo "no per-sweep tables written" >&2; exit 1; }

# the experiment list comes from the reference: the crystal models are the
# same ones the predictions were made against
args=()
for refl in "${refls[@]}"; do
    sweep=$(basename "$refl" .refl)
    args+=("$DATA_DIR/dials_reference/$sweep/integrated.expt" "$refl")
done

echo "===== dials.scale over ${#refls[@]} sweep(s)"
( cd "$work" && dials.scale "${args[@]}" anomalous=True > scale.log 2>&1 )
( cd "$work" && dials.merge scaled.expt scaled.refl anomalous=True \
      output.html=merged.html output.log=merged.log output.mtz=merged.mtz \
      > merge_run.log 2>&1 )

python "$INTEGRATOR_ROOT/scripts/mono/emit_merging_stats.py" "$work/merged.html"

echo "===== refine and find peaks"
python "$INTEGRATOR_ROOT/scripts/sbgrid/refine.py" \
    --mtz "$work/merged.mtz" --data-dir "$DATA_DIR"

echo "===== done: $work"
