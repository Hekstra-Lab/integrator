#!/bin/bash
# Submit all 48 valid model+loss+profile+surrogate combinations.
#
# Usage:
#   bash scripts/submit_all.sh [wb_project]
#
# Configs must exist in configs/comparison/ (run generate_comparison_configs.py first).

set -euo pipefail

WB_PROJECT=${1:-"ModelComparison"}
CONFIG_DIR="configs/comparison"
SCRIPT="scripts/train_job.sh"

SURROGATES=(gammaA gammaB gammaC gammaD folded_normal log_normal)

# Physical gaussian is 2D-only — skip for 3D configs
SKIP_PHYSICAL=true

count=0
for config in "$CONFIG_DIR"/*.yaml; do
    config_label=$(basename "$config" .yaml)

    # Skip physical gaussian (incompatible with 3D data)
    if $SKIP_PHYSICAL && [[ "$config_label" == *"physical"* ]]; then
        echo "SKIP  $config_label  (physical gaussian is 2D-only)"
        continue
    fi

    for surr in "${SURROGATES[@]}"; do
        echo "SUBMIT  ${config_label} | ${surr}"
        sbatch "$SCRIPT" "$config" "$surr" "$WB_PROJECT"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted $count jobs to SLURM"
echo "Monitor with: squeue -u \$USER"
