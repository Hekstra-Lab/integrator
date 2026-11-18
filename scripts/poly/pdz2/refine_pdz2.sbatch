#!/bin/bash
#SBATCH -J pdz2_refine
#SBATCH -p sapphire,shared
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/refine_off/logs/%x_%j.out
# Refine the deposited model against each arm off-state merge. Identical protocol and
# identical held-out reflections across arms, so the only variable is integration.
set -uo pipefail
B=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/refine_off
source /n/lab_storage/hekstra_lab/people/aldama/software/phenix2/phenix_env.sh
for ARM in laue_r1 laue_r2 geom_both_r1 geom_both_r2 ml_clamp_r1 ml_clamp_r2; do
  D=$B/run_$ARM; rm -rf $D; mkdir -p $D; cd $D
  echo "==== $ARM"
  phenix.refine \
    $B/refine_in_${ARM}.mtz \
    $B/model_recelled.pdb \
    refinement.main.number_of_macro_cycles=3 \
    refinement.input.xray_data.labels=F,SigF \
    refinement.input.xray_data.r_free_flags.label=FreeR_flag \
    output.prefix=refined \
    output.overwrite=True > $D/phenix.log 2>&1
  grep -hE "^Final R-work|^Start R-work" $D/refined_001.log 2>/dev/null | tail -2
done
echo "REFINE_ALL_DONE"
