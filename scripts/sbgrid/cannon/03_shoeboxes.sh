#!/bin/bash
# Choose the shoebox window from the boxes DIALS used, then cut them.
#
# The window is chosen, not configured: the smallest odd box covering
# --coverage of the integrated reflections on each axis.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
source "$MAMBA_SH"
micromamba activate "$ENV_MAIN"

python "$INTEGRATOR_ROOT/scripts/sbgrid/shoebox_size.py" \
    --data-dir "$DATA_DIR" --coverage "${COVERAGE:-0.99}" --update-card

read -r D H W < <(python -c "
import json, pathlib
c = json.loads(pathlib.Path('$DATA_DIR/dataset_card.json').read_text())['shoebox']
print(c['d'], c['h'], c['w'])
")
echo "===== cutting ${D}x${H}x${W} shoeboxes"

# one sweep at a time, as DIALS integrated them
for expt in "$DATA_DIR"/dials_reference/*/integrated.expt; do
    sweep="$(basename "$(dirname "$expt")")"
    integrator.make_shoeboxes \
        --data-dir "$(dirname "$expt")" \
        --refl integrated.refl --expt integrated.expt \
        --out-dir "$OUT/shoeboxes/$sweep" \
        --d "$D" --h "$H" --w "$W"
done
