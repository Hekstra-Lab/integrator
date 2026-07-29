#!/usr/bin/env bash
#SBATCH --account=lcls:prjlumine22
#SBATCH --partition=milano
#SBATCH --qos=normal
#SBATCH --constraint=OS_VER:8.6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --time=0-8:00
#SBATCH --job-name=merge_mfx_boxes
#SBATCH --output=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/merge_allruns_024_rg101_%j.out
#SBATCH --error=/sdf/scratch/lcls/ds/prj/prjlumine22/scratch/thaoh/mfx101555026_cctbx/logs/merge_allruns_024_rg101_%j.err

set -eo pipefail

source /sdf/group/lcls/ds/tools/cctbx/psana2_setup.sh
export PYTHONPATH=/sdf/home/t/thaoh/s3df_practice/integrator/src:$PYTHONPATH

cd /sdf/home/t/thaoh/s3df_practice/integrator

time dials.python workflows/mfx101555026/scripts/merge_allruns_024_rg101_shoebox_chunks.py
