#!/bin/bash
# Re-merge one epoch's predictions under several variance models.
#
# Holds the intensities fixed and changes only the merge weights, so a change
# in CC-half is attributable to the error bars rather than to the intensities.
#
#   ./06_variance_test.sh <run-dir> <epoch> [variance ...]
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
source "$MAMBA_SH"
micromamba activate "$ENV_MAIN"

run_dir="${1:?Usage: ./06_variance_test.sh <run-dir> <epoch> [variance ...]}"
epoch="${2:?epoch}"
shift 2
variances=("${@:-poisson}")

pred_dir=$(python - "$run_dir" <<'PY'
import sys, yaml, pathlib
print(yaml.safe_load((pathlib.Path(sys.argv[1]) / "run_paths.yaml").read_text())["predictions_dir"])
PY
)
epoch_dir="$pred_dir/$(printf 'epoch_%04d' "$epoch")"

for variance in "${variances[@]}"; do
    echo "##### variance model: $variance"
    python "$INTEGRATOR_ROOT/scripts/sbgrid/remerge_variance.py" \
        --pred-dir "$epoch_dir" --dataset-dir "$OUT/dataset" --variance "$variance"

    work="$epoch_dir/scaled_$variance"
    mkdir -p "$work"
    args=()
    for refl in "$epoch_dir/per_sweep_$variance"/*.refl; do
        sweep=$(basename "$refl" .refl)
        args+=("$DATA_DIR/dials_reference/$sweep/integrated.expt" "$refl")
    done
    ( cd "$work" && dials.scale "${args[@]}" anomalous=True > scale.log 2>&1 )
    ( cd "$work" && dials.merge scaled.expt scaled.refl anomalous=True \
          output.html=merged.html output.log=merged.log output.mtz=merged.mtz \
          > merge_run.log 2>&1 )
    python "$INTEGRATOR_ROOT/scripts/mono/emit_merging_stats.py" "$work/merged.html"
    echo "  -> $work/merging_stats.csv"
done
