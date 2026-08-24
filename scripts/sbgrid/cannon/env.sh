# Shared settings for one SBGrid dataset. Sourced by every 0*.sh here.
#
# Copy this directory into the dataset folder so the scripts that produced a
# result sit beside it. Everything large -- images, DIALS output, shoeboxes,
# checkpoints -- stays on scratch.

export SBGRID_ID="${SBGRID_ID:?set SBGRID_ID, e.g. SBGRID_ID=821}"
export PDB_ID="${PDB_ID:-}"          # the entry this dataset produced, if any

export INTEGRATOR_ROOT="${INTEGRATOR_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/software/integrator}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/n/netscratch/hekstra_lab/Lab/laldama}"
export DATA_DIR="${DATA_DIR:-$SCRATCH_ROOT/sbgrid/$SBGRID_ID}"
export KIT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# where the integrator's own runs go, kept apart from the reference
export OUT="${OUT:-$DATA_DIR/integrator}"

export MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
# integrator-cuda-dev throughout: the standalone dials env predates the
# storage migration and its console scripts carry dead shebangs
export ENV_MAIN="${ENV_MAIN:-integrator-cuda-dev}"

export WB_PROJECT="${WB_PROJECT:-sbgrid_$SBGRID_ID}"
export WB_SAVE_DIR="${WB_SAVE_DIR:-$OUT/wandb_logs}"

# CPU partitions: `test` has 112-core nodes and allows 5 submitted jobs per
# user, against gpu_test's 2. One partition per job -- a comma-separated list
# is rejected because they sit under different QOS.
export CPU_PARTITION="${CPU_PARTITION:-test}"
export GPU_PARTITION="${GPU_PARTITION:-gpu_test}"
