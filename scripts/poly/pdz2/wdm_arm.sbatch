#!/bin/bash
#SBATCH -J pdz2_wdm
#SBATCH -p shared,serial_requeue
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 0-01:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/logs/wdm_%x_%j.out
set -euo pipefail
TREE=${TREE:?tree path}
REF=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
# wdm.py reads 5e1y_phases.mtz from the tree root and writes into gain_0,3_from_stills/
ln -sf $REF/5e1y_phases.mtz $TREE/5e1y_phases.mtz
cp -n $REF/scripts/wdm.py $TREE/scripts/ 2>/dev/null || true
cd $TREE && micromamba run -n integrator-cuda-dev python scripts/wdm.py 0.3 2>&1 | tail -6
ls -la $TREE/gain_0,3_from_stills/wdm_rescut.mtz
echo "WDM_DONE"
