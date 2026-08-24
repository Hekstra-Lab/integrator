#!/bin/bash
# Multi-model comparison figures: loss curves, model-vs-DIALS scatters,
# anomalous peaks, refinement R-factors.
#
# Separate from 04_figures.sh, which renders the per-run training figures
# (tracked shoeboxes, basis, latent). This one compares runs to each other
# and to the references in plot_cfg.yaml.
#
#   ./05_plots.sh                          # uses ./plot_cfg.yaml
#   ./05_plots.sh my_cfg.yaml              # a different manifest
#   OUT_DIR=/somewhere ./05_plots.sh
#
# A missing input makes only that step warn and skip: no peaks.csv yet means
# no peak figure, the loss curves still render.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

source "$MAMBA_SH"
micromamba activate "$ENV_TRAIN"

RUNNER=python INTEGRATOR_ROOT="$INTEGRATOR_ROOT" \
    "$INTEGRATOR_ROOT/scripts/make_figures.sh" \
    --plot-cfg "${1:-$OUT/plot_cfg.yaml}" \
    --out-dir "${OUT_DIR:-$OUT/figures_comparison}"
