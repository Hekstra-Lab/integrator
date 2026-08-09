#!/usr/bin/env bash
#SBATCH --job-name=scale_merge_0_15
#SBATCH --partition=milano
#SBATCH --account=lcls:prjlumine22
#SBATCH --qos=normal
#SBATCH --constraint=OS_VER:8.6
#SBATCH -N 1
#SBATCH --ntasks-per-node=120
#SBATCH --time=08:00:00
#SBATCH --output=scale_merge_all_dmin0_dmin15_%j.out
#SBATCH --error=scale_merge_all_dmin0_dmin15_%j.err

set -eo pipefail

export PS1="${PS1:-}"

cd /sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/combinedmask_scale_merge_phil

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh

echo "Starting scale dmin 0"
mpirun cctbx.xfel.merge scale_all_r269_r289__dmin0.phil

echo "Starting merge dmin 0"
mpirun cctbx.xfel.merge merge_all_r269_r289__dmin0.phil

echo "Starting scale dmin 1.5"
mpirun cctbx.xfel.merge scale_all_r269_r289__dmin15.phil

echo "Starting merge dmin 1.5"
mpirun cctbx.xfel.merge merge_all_r269_r289__dmin15.phil

echo "All done"
