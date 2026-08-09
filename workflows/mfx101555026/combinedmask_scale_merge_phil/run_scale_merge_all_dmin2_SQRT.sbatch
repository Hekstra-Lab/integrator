#!/usr/bin/env bash
#SBATCH --partition=milano
#SBATCH --account=lcls:prjlumine22
#SBATCH --qos=normal
#SBATCH --constraint=OS_VER:8.6
#SBATCH -N 1
#SBATCH --ntasks-per-node=120
#SBATCH --time=08:00:00
#SBATCH --job-name=predicted_sqrt_dmin2
#SBATCH --output=pred_asinh_dmin2_%j.out
#SBATCH --error=pred_asinh_dmin2_%j.err

set -eo pipefail
export PS1="${PS1:-}"

cd /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/combinedmask_scale_merge_phil

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

echo "Starting predicted SQRT scale dmin 2.0"
mpirun cctbx.xfel.merge PREDICTED_scale_all_r269_r289__dmin2_SQRT.phil

echo "Starting predicted SQRT merge dmin 2.0"
mpirun cctbx.xfel.merge PREDICTED_merge_all_r269_r289__dmin2_SQRT.phil

echo "All done for predicted SQRT dmin 2.0"