#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-08:00
#SBATCH -o nb_screen_%A_%a.out
#SBATCH -e nb_screen_%A_%a.err
#SBATCH --array=0-9
#
# Negative-Binomial dispersion SCREEN as a SLURM job array: one array task per arm.
# Each arm holds the NB dispersion r FIXED (nb_learn_dispersion=false) and varies
# nb_dispersion_init, so the array traces the ELBO along the overdispersion axis
#   Var = mean + mean^2 / r,   r -> inf  ==  Poisson.
# Arm 0 is the Poisson baseline (r = inf); the last arm learns r, for reference.
#
# The array range MUST match the arm count: 1 (poisson) + len(DISPERSIONS) + 1
# (learned). Default DISPERSIONS has 8 values -> 10 arms -> --array=0-9. If you
# change DISPERSIONS, update --array above to 0-$((len+1)).
#
# PREREQ: set INTEGRATOR_ROOT (repo root) in your shell rc, e.g.
#     export INTEGRATOR_ROOT=/n/lab_storage/hekstra_lab/people/aldama/software/integrator
#
# SUBMIT:  sbatch scripts/ablation_likelihood/dispersion_screen.slurm
#   subset:      sbatch --array=0,5 scripts/ablation_likelihood/dispersion_screen.slurm
#   more epochs: EPOCHS=50 sbatch scripts/ablation_likelihood/dispersion_screen.slurm
#   w&b:         WB_PROJECT=nb_screen sbatch scripts/ablation_likelihood/dispersion_screen.slurm
#   custom grid: DISPERSIONS="1 4 16 64" sbatch --array=0-5 scripts/ablation_likelihood/dispersion_screen.slurm

set -euo pipefail
export TQDM_DISABLE=1

# ── Knobs (override via env: EPOCHS=... OUT=... DISPERSIONS=... sbatch ...) ────
INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set (add to ~/.bashrc).}"
BASE="${BASE:-$INTEGRATOR_ROOT/configs/ablation_likelihood/hierarchical_nbinom.yaml}"
OUT="${OUT:-$PWD/nb_screen}"                 # parent dir for the per-arm run dirs
EPOCHS="${EPOCHS:-25}"
SCOPE="${SCOPE:-global}"                      # global (one r) | per_bin (r per bin)
DATA_DIR="${DATA_DIR:-}"                       # override config data_dir, or leave blank
WB_PROJECT="${WB_PROJECT:-}"                   # set to enable W&B logging
PATCH="$INTEGRATOR_ROOT/scripts/ablation_likelihood/make_screen_config.py"

read -r -a DISP <<< "${DISPERSIONS:-0.5 1 2 5 10 25 50 100}"

# ── Build the arm list: poisson baseline, fixed-r grid, learned-r reference ───
ARMS=("poisson")
for r in "${DISP[@]}"; do ARMS+=("nb_fixed:$r"); done
ARMS+=("nb_learned")

n_arms=${#ARMS[@]}
if (( SLURM_ARRAY_TASK_ID >= n_arms )); then
    echo "ERROR: task $SLURM_ARRAY_TASK_ID >= $n_arms arms. Fix --array (0-$((n_arms-1)))." >&2
    exit 1
fi
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"

# ── Environment ───────────────────────────────────────────────────────────────
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
micromamba activate integrator
cd "$INTEGRATOR_ROOT"

# ── Translate the arm into (tag, patcher flags) ───────────────────────────────
kind="${ARM%%:*}"
value="${ARM#*:}"
case "$kind" in
    poisson)
        tag="poisson"
        patch_flags=(--likelihood poisson)
        ;;
    nb_fixed)
        tag="nb_r${value//./p}"                            # 0.5 -> nb_r0p5
        patch_flags=(--likelihood negative_binomial --dispersion "$value" --scope "$SCOPE")
        ;;
    nb_learned)
        tag="nb_learned"
        patch_flags=(--likelihood negative_binomial --dispersion 10 --scope "$SCOPE" --learn-dispersion)
        ;;
    *)
        echo "ERROR: unknown arm '$ARM'." >&2; exit 1 ;;
esac

run_dir="$OUT/$tag"
mkdir -p "$run_dir"
config="$run_dir/screen_config.yaml"

echo "host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-?}  task $SLURM_ARRAY_TASK_ID/$((n_arms-1))  arm: $ARM  tag: $tag"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Materialize this arm's config ─────────────────────────────────────────────
python "$PATCH" --base "$BASE" --out "$config" "${patch_flags[@]}"
echo "===== patched config -> $config ====="

# ── Assemble the training command ─────────────────────────────────────────────
# Invoke via the module path, not the `integrator.train` console script: pip's
# entry-point wrapper hardcodes an absolute interpreter shebang, which breaks when
# the micromamba env is relocated (e.g. the /n/hekstra_lab -> /n/lab_storage move).
# `python -m` uses the activated env's interpreter, so it survives the migration.
cmd=(python -m integrator.cli.train -v --config "$config" --run-dir "$run_dir" --max-epochs "$EPOCHS")
[[ -n "$DATA_DIR" ]] && cmd+=(--data-dir "$DATA_DIR")

if [[ -n "$WB_PROJECT" ]]; then
    cmd+=(--wb-project "$WB_PROJECT" --tags nb_screen "$tag" "r=${value}")
else
    # No W&B: output_root == run_dir, so the checkpoint path is deterministic and
    # a resubmitted array resumes from the latest checkpoint of each arm.
    ckpt="$run_dir/files/checkpoints/last.ckpt"
    [[ -e "$ckpt" ]] && { echo "resuming from $ckpt"; cmd+=(--ckpt-path "$ckpt"); }
fi

echo "===== ${cmd[*]} ====="
"${cmd[@]}"

echo "done task $SLURM_ARRAY_TASK_ID ($tag)"
