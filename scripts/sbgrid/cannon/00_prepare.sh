#!/bin/bash
# Download the dataset and work out how to process it. Minutes, no GPU.
#
#   SBGRID_ID=821 PDB_ID=7LVC ./00_prepare.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
source "$MAMBA_SH"
micromamba activate "$ENV_MAIN"

S="$INTEGRATOR_ROOT/scripts/sbgrid"

if [[ ! -d "$DATA_DIR" || -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]]; then
    echo "===== downloading SBGrid $SBGRID_ID"
    SBGRID_ID="$SBGRID_ID" sbatch -W "$S/fetch.slurm"
fi

# the depositors' own recipe, where they published one: masks and beam-centre
# corrections that the image headers do not carry
python "$S/bundle.py" --id "$SBGRID_ID" --out-dir "$DATA_DIR"

[[ -n "$PDB_ID" ]] && python "$S/reference_stats.py" --pdb "$PDB_ID" --out-dir "$DATA_DIR"

python "$S/characterize.py" --data-dir "$DATA_DIR" ${PDB_ID:+--pdb "$PDB_ID"}
