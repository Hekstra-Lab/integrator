#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-10:00
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

# Usage:
#   sbatch train_job.sh <config> <surrogate> [wb_project]
#
# <config> resolution order (first match wins):
#   1. Absolute path or exists relative to cwd       (old behavior)
#   2. $INTEGRATOR_CONFIGS/<config>                   (env var if set)
#   3. $INTEGRATOR_ROOT/configs/<config>              (sibling to the code)
#
# Examples (all equivalent after resolution):
#   sbatch train_job.sh /abs/path/to/foo.yaml gammaB
#   sbatch train_job.sh hierC_wilson_learned_nowarm_gammaB.yaml gammaB
#   sbatch train_job.sh wilson_comparison/hierC_learned.yaml gammaB
#
# Set INTEGRATOR_ROOT in your shell rc (e.g. ~/.bashrc):
#     export INTEGRATOR_ROOT=/path/to/integrator

set -euo pipefail
export TQDM_DISABLE=1

# ── Args ──────────────────────────────────────────────────────────────
config_arg=${1:?  "Usage: train_job.sh <config> <surrogate> [wb_project]"}
surrogate=${2:?   "Usage: train_job.sh <config> <surrogate> [wb_project]"}
wb_project=${3:-"ModelComparison"}

# ── Integrator repo location ──────────────────────────────────────────
INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set (add to ~/.bashrc).}"
INTEGRATOR_CONFIGS="${INTEGRATOR_CONFIGS:-$INTEGRATOR_ROOT/configs}"

# ── Resolve config to an absolute path ────────────────────────────────
if [[ -f "$config_arg" ]]; then
    config="$(realpath "$config_arg")"
elif [[ -f "$INTEGRATOR_CONFIGS/$config_arg" ]]; then
    config="$(realpath "$INTEGRATOR_CONFIGS/$config_arg")"
else
    echo "ERROR: config '$config_arg' not found." >&2
    echo "  Checked:  $(realpath -m "$config_arg" 2>/dev/null || echo "$config_arg")" >&2
    echo "  Checked:  $INTEGRATOR_CONFIGS/$config_arg" >&2
    echo "  (Set INTEGRATOR_CONFIGS or INTEGRATOR_ROOT to override.)" >&2
    exit 1
fi

echo "===== Resolved config: $config ====="

# ── Derive a human-readable label from the YAML ───────────────────────
read -r model_name loss_name profile_surr i_prior bg_prior < <(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('$config'))
model   = cfg['integrator']['name']
loss    = cfg['loss']['name']
profile = cfg['surrogates']['qp']['name']
pi  = (cfg['loss'].get('args',{}).get('pi_cfg',{})  or {}).get('name','none')
pbg = (cfg['loss'].get('args',{}).get('pbg_cfg',{}) or {}).get('name','none')
short = {'exponential':'expo', 'gamma':'gamma', 'half_cauchy':'hc',
         'log_normal':'ln', 'none':'none'}
print(model, loss, profile, short.get(pi,pi), short.get(pbg,pbg))
")
config_label="${model_name}_${loss_name}_${profile_surr}"
run_label="${config_label}_${surrogate}_pi-${i_prior}_pbg-${bg_prior}"

# refltorch scripts for dials/phenix post-processing
refltorch_dir=/n/hekstra_lab/people/aldama/refltorch/scripts/dials_output

# ── Environment ───────────────────────────────────────────────────────
source /n/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
micromamba activate integrator

# ── Run directory (created in CWD = your comparison dir) ──────────────
run_dir="${run_label}_${SLURM_JOB_ID}"
mkdir -p "$run_dir"

echo "===== Config:    $config ====="
echo "===== Surrogate: $surrogate ====="
echo "===== Run dir:   $(realpath "$run_dir") ====="

# ── Train ─────────────────────────────────────────────────────────────
echo "===== Starting integrator.train ====="
integrator.train -v \
    --config "$config" \
    --wb-project "$wb_project" \
    --qbg "$surrogate" \
    --qi "$surrogate" \
    --run-dir "$run_dir" \
    --tags "$config_label" "$surrogate" "pi-${i_prior}" "pbg-${bg_prior}"

# ── Predict ───────────────────────────────────────────────────────────
echo "===== Starting integrator.pred ====="
integrator.pred -v \
    --run-dir "$run_dir" \
    --write-refl

# ── DIALS/Phenix post-processing ─────────────────────────────────────
echo "===== Starting DIALS-Phenix Parallel Processing Setup ====="

micromamba deactivate
micromamba activate refltorch

log_dir="${run_dir}/dials_phenix_logs"
mkdir -p "$log_dir"

python "$refltorch_dir/create_config.py" \
    --run-dir "$run_dir"

python "$refltorch_dir/submit_jobs.py" \
    --run-dir "$run_dir" \
    --log-dir "$log_dir" \
    --script-dir "$refltorch_dir"

echo "===== Done ====="
