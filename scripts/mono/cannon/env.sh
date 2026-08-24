# Shared settings for the HEWL monochromatic (rotation) run. Sourced by every 0*.sh here.
# Mirror of scripts/poly/cannon/env.sh; the mono arm swaps the config, the W&B
# project, and the downstream tool (DIALS scale+merge+phenix instead of careless).

# Repo checkout
export INTEGRATOR_ROOT="${INTEGRATOR_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/software/integrator}"

# micromamba hook + env (integrator-cuda-dev holds integrator.train/.predict,
# DIALS, phenix is sourced per-step inside process_single.py)
export MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
export ENV_TRAIN="${ENV_TRAIN:-integrator-cuda-dev}"   # holds integrator.train / .predict

# Outputs go to scratch, never next to the scripts. A run writes tens of GB
# of checkpoints, predictions and MTZs; /n/holylabs is a small per-lab
# allocation and filling it makes every sbatch fail at launch, because SLURM
# cannot create the job's .out file. /n/netscratch is the large volume.
export SCRATCH_ROOT="${SCRATCH_ROOT:-/n/netscratch/hekstra_lab/Lab/laldama}"
export OUT="${OUT:-$SCRATCH_ROOT/integrator_runs}"
mkdir -p "$OUT"

# where these scripts live, for the wrappers that need to find each other
export KIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Configs (resolve through KIT_DIR / INTEGRATOR_ROOT, never through OUT)
export CONFIG="${CONFIG:-$INTEGRATOR_ROOT/configs/mono/hewl9b7c_mono.yaml}"
# Mono downstream config: the DIALS/phenix post-processing settings (analog of
# poly's PIPELINE_CFG). Consumed by post_config.py to write dials_phenix_cfg.yaml.
export PROCESS_CFG="${PROCESS_CFG:-/n/lab_storage/hekstra_lab/people/aldama/integrator_data/hewl_9b7c/scaling_dataset/processing_config.yaml}"

# W&B: on by default. Set WB_PROJECT="" to fall back to local-only logging.
# The local RunLogger writes metrics.csv, the loss curves, and every figure
# dump into the run dir either way, so W&B is additive.
#
# With W&B on, the output root moves under <WB_SAVE_DIR>/wandb/run-<id>/ —
# checkpoints, plots, predictions, and figures all live there, and the run
# dir keeps only run_paths.yaml as the handle. Everything downstream
# (03_pipeline.sh, 04_figures.sh) resolves paths from that file, so the move
# is transparent.
export WB_PROJECT="${WB_PROJECT:-hewl_mono}"
export WB_SAVE_DIR="${WB_SAVE_DIR:-$OUT/wandb_logs}"
