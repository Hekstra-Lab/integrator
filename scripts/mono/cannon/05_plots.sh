#!/bin/bash
# Multi-model comparison figures: loss curves, model-vs-DIALS scatters,
# anomalous peaks, refinement R-factors, merging statistics.
#
# Separate from 04_figures.sh, which renders the per-run training figures
# (tracked shoeboxes, basis, latent). This one compares runs to each other
# and to the references in plot_cfg.yaml. It calls the shared driver
# scripts/make_figures.sh, which is the same one the poly arm uses — so a
# single plot_cfg listing mono and poly run dirs produces side-by-side figures.
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
    --plot-cfg "${1:-$KIT_DIR/plot_cfg.yaml}" \
    --out-dir "${OUT_DIR:-$OUT/figures_comparison}"
