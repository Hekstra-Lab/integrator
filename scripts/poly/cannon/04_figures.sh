#!/bin/bash
# Render the training figures and bundle them into one HTML page.
# Works whether the run collected dumps live (--figures) or not: without
# them this replays the run's checkpoints instead.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

run_dir="${1:?Usage: ./04_figures.sh <run-dir>}"

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

python "$INTEGRATOR_ROOT/scripts/make_training_figures.py" \
    --run-dir "$run_dir" -v

fig_dir="$(python - "$run_dir" <<'PY'
import sys, yaml, pathlib
meta = yaml.safe_load((pathlib.Path(sys.argv[1]) / "run_paths.yaml").read_text())
print(meta.get("figures_dir") or (pathlib.Path(meta["output_root"]) / "figures"))
PY
)"

python "$INTEGRATOR_ROOT/scripts/make_figure_report.py" \
    --fig-dir "$fig_dir" \
    --title "HEWL Laue — $(basename "$run_dir")" -v
