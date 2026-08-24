#!/bin/bash
# One command for the whole thing: preflight -> train -> predict/careless/
# phenix/peaks -> per-run figures -> comparison plots.
#
# Each stage is its own SLURM job, chained with --dependency=afterok, because
# training and careless each want their own allocation for hours and a single
# job would hold a GPU idle through phenix. Submitting the chain up front
# means one command, everything queued, each stage starting when its input
# exists, and a failure stopping the rest instead of cascading garbage.
#
#   ./run_all.sh                          # the full chain, new run
#   ./run_all.sh --from pipeline          # training already done
#   ./run_all.sh --from figures --run-name hewl1118_g5000
#   ./run_all.sh --only plots
#   ./run_all.sh --dry-run                # print the sbatch calls, submit nothing
#
# Options:
#   --run-name NAME   run directory under this one (default: hewl_<date>)
#   --from STAGE      start here (train | pipeline | figures | plots)
#   --to STAGE        stop after here
#   --only STAGE      exactly one stage
#   --epochs N        training epochs (default 40)
#   --config FILE     training config (default configs/poly/hewl1118_poly.yaml)
#   --pipeline-cfg F  downstream config (default ./poly_pipeline_cfg.yaml)
#   --no-preflight    skip the pre-submission validation
#   --dry-run         print, submit nothing

set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

STAGES=(train pipeline figures plots)
FROM=train
TO=plots
EPOCHS="${EPOCHS:-40}"
DRY=0
PREFLIGHT=1
RUN_NAME="${RUN_NAME:-hewl_$(date +%Y%m%d)}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)      RUN_NAME="$2"; shift 2;;
        --from)          FROM="$2"; shift 2;;
        --to)            TO="$2"; shift 2;;
        --only)          FROM="$2"; TO="$2"; shift 2;;
        --epochs)        EPOCHS="$2"; shift 2;;
        --config)        CONFIG="$2"; shift 2;;
        --pipeline-cfg)  PIPELINE_CFG="$2"; shift 2;;
        --no-preflight)  PREFLIGHT=0; shift;;
        --dry-run)       DRY=1; shift;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1;;
    esac
done

index_of() {
    local want="$1" i=0
    for s in "${STAGES[@]}"; do
        [[ "$s" == "$want" ]] && { echo "$i"; return; }
        i=$((i + 1))
    done
    echo "ERROR: unknown stage '$want' (want one of ${STAGES[*]})" >&2
    exit 1
}
from_i=$(index_of "$FROM")
to_i=$(index_of "$TO")
[[ "$from_i" -le "$to_i" ]] || { echo "ERROR: --from is after --to" >&2; exit 1; }

run_dir="$OUT/$RUN_NAME"
echo "run name : $RUN_NAME"
echo "run dir  : $run_dir"
echo "stages   : ${STAGES[*]:$from_i:$((to_i - from_i + 1))}"
echo "config   : $CONFIG"
[[ -n "${WB_PROJECT:-}" ]] && echo "wandb    : $WB_PROJECT"
echo

# Validate before anything is queued: a bad path should cost seconds, not a
# place in the queue followed by an immediate failure.
if [[ "$PREFLIGHT" == "1" && "$DRY" == "0" && "$from_i" -eq 0 ]]; then
    source "$MAMBA_SH"
    micromamba activate "$ENV_TRAIN"
    python "$INTEGRATOR_ROOT/scripts/preflight.py" \
        --config "$CONFIG" --pipeline-cfg "$PIPELINE_CFG"
    echo
fi

submit() {  # submit <name> <dependency-or-empty> <sbatch args...>
    local name="$1" dep="$2"; shift 2
    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")
    if [[ "$DRY" == "1" ]]; then
        echo "[$name] sbatch ${depflag[*]} $*" >&2
        echo "<${name}_jobid>"
        return
    fi
    local id
    id=$(sbatch --parsable "${depflag[@]}" "$@")
    echo "  $name -> job $id${dep:+ (after $dep)}" >&2
    echo "$id"
}

# gpu_test caps a one-GPU job at under 8 cores and under ~64 GB per GPU.
GPU_ARGS=(-p "${PARTITION:-gpu_test}" --gres=gpu:1 -t "${WALLTIME:-12:00:00}" -c 7 --mem=60G)
CPU_ARGS=(-p "${CPU_PARTITION:-shared}" -t 02:00:00 -c 4 --mem=32G)

dep=""
declare -A jobs

if [[ "$from_i" -le 0 && "$to_i" -ge 0 ]]; then
    jobs[train]=$(RUN_NAME="$RUN_NAME" EPOCHS="$EPOCHS" CONFIG="$CONFIG" \
        NUM_WORKERS="${NUM_WORKERS:-6}" \
        submit train "$dep" "${GPU_ARGS[@]}" ./02b_train_requeue.sh)
    dep="${jobs[train]}"
fi

if [[ "$from_i" -le 1 && "$to_i" -ge 1 ]]; then
    jobs[pipeline]=$(RUN_DIR="$run_dir" PIPELINE_CFG="$PIPELINE_CFG" \
        STEPS="${STEPS:-predict,careless,phenix,peaks}" \
        submit pipeline "$dep" "${GPU_ARGS[@]}" \
            "$INTEGRATOR_ROOT/scripts/poly/pipeline.slurm")
    dep="${jobs[pipeline]}"
fi

if [[ "$from_i" -le 2 && "$to_i" -ge 2 ]]; then
    jobs[figures]=$(submit figures "$dep" "${CPU_ARGS[@]}" \
        -J figures -o "figures_%j.out" -e "figures_%j.err" \
        --wrap "cd $OUT && ./04_figures.sh $run_dir")
    dep="${jobs[figures]}"
fi

if [[ "$from_i" -le 3 && "$to_i" -ge 3 ]]; then
    jobs[plots]=$(submit plots "$dep" "${CPU_ARGS[@]}" \
        -J plots -o "plots_%j.out" -e "plots_%j.err" \
        --wrap "cd $OUT && ./05_plots.sh")
fi

echo
if [[ "$DRY" == "1" ]]; then
    echo "dry run: nothing submitted"
else
    echo "chain submitted. watch with:  squeue -u \$USER"
    echo "cancel the rest with:        scancel ${jobs[*]}"
fi
