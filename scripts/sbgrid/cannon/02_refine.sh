#!/bin/bash
# phenix.refine plus the anomalous peak search, on whichever merged MTZ is
# named. Used for the reference and for the integrator's own merge, with the
# same model and parameters so the numbers compare.
#
#   ./02_refine.sh                      # the DIALS reference
#   ./02_refine.sh <some other>.mtz     # any other merge
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
source "$MAMBA_SH"
micromamba activate "$ENV_MAIN"

mtz="${1:-$DATA_DIR/dials_reference/merged.mtz}"
python "$INTEGRATOR_ROOT/scripts/sbgrid/refine.py" \
    --mtz "$mtz" --data-dir "$DATA_DIR" ${OUT_DIR:+--out-dir "$OUT_DIR"}
