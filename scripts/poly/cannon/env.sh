# Shared settings for the HEWL Laue run. Sourced by every 0*.sh here.
# Verified live 2026-08-23: every path and env below exists.

# Repo checkout (current with origin as of 2026-08-23 14:12)
export INTEGRATOR_ROOT="${INTEGRATOR_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/software/integrator}"

# micromamba hook + envs
export MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
export ENV_TRAIN="${ENV_TRAIN:-integrator-cuda-dev}"   # holds integrator.train / .predict
export ENV_DRIVER="${ENV_DRIVER:-integrator-cuda-dev}" # runs run_pipeline.py
export ENV_CARELESS="${ENV_CARELESS:-crls}"            # careless + careless.ccanom

# This directory is the run root: run dirs land next to these scripts.
export OUT="${OUT:-$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )}"

# Configs
export CONFIG="${CONFIG:-$INTEGRATOR_ROOT/configs/poly/hewl1118_poly.yaml}"
export PIPELINE_CFG="${PIPELINE_CFG:-$OUT/poly_pipeline_cfg.yaml}"

# W&B: on by default. Set WB_PROJECT="" to fall back to local-only logging.
# The local RunLogger writes metrics.csv, the loss curves, and every figure
# dump into the run dir either way, so W&B is additive.
#
# With W&B on, the output root moves under <WB_SAVE_DIR>/wandb/run-<id>/ —
# checkpoints, plots, predictions, and figures all live there, and the run
# dir keeps only run_paths.yaml as the handle. Everything downstream
# (03_pipeline.sh, 04_figures.sh) resolves paths from that file, so the move
# is transparent.
export WB_PROJECT="${WB_PROJECT:-hewl_laue}"
export WB_SAVE_DIR="${WB_SAVE_DIR:-$OUT/lightning_logs}"
