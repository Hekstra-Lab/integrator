#!/bin/bash
# laue-dials on the three mosaicity levels, reproducing the settings used for
# hkl_data (mos_spread 0.1) and changing only the input directory.
#
#   shot1  mos_spread 0.05    shot2  0.25    shot3  0.50
#
# Every step is CPU. laue.predict is pinned to one BLAS thread inside
# laue_process.sh because it forks a Pool after training a scipy KDE.
#SBATCH -J moslaue
#SBATCH -p sapphire
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0-12:00
#SBATCH -a 0-2
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/ablate_logs/%x_%A_%a.log
#SBATCH -e /n/netscratch/hekstra_lab/Lab/laldama/ablate_logs/%x_%A_%a.log
set -uo pipefail
S=/n/netscratch/hekstra_lab/Lab/laldama
SRC=/n/holylabs/hekstra_lab/Users/laldama/data/simdhfr/data/mosaicity
SHOTS=(shot1 shot2 shot3)
MOS=(005 025 050)
i=$SLURM_ARRAY_TASK_ID
SH=${SHOTS[$i]}; M=${MOS[$i]}
OUT=$S/sim_mos/mos$M
mkdir -p "$OUT"

export DATA_DIR="$SRC/$SH"
export IMG_TEMPLATE="$SRC/$SH/shot_1_00###.cbf"
export OUT
export TAG="simdhfr_mos$M"
# settings recovered from sim/laue180, the run on hkl_data
export OSC="0,1"
export BEAM_WL=1.04
export IMAGE_RANGE="1,180"
export SG=19
export CELL="33.15729336076872,43.9678623861119,95.5472958310359,90,90,90"
export LAM_MIN=1.0
export LAM_MAX=1.2
export GAIN=1
export DMIN=1.4
export NPROC=16
export ENV_LD=integrator-cuda-dev

echo "=== mosaicity $M ($SH) -> $OUT ==="
bash $S/integrator_plumbing/scripts/poly/laue_process.sh
echo "=== exit $? ==="
ls -la "$OUT"/*.mtz 2>/dev/null | awk '{print $5, $9}'
