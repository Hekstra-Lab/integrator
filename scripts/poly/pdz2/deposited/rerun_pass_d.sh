#!/bin/bash
# Re-run pass d with start angle 0 (start 92 hits the laue.sequence_to_stills
# off-by-one-step bug, ~15 px RMSD; start 0 verified to give the identical Miller
# setting), then re-chain expt_concat -> careless -> stats.
set -euo pipefail
cd /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref
g=0.3
jo=$(sbatch --parsable scripts/one_pass-from_off.sh d off 0 2 $g x,y,z); echo "d off -> $jo"
ds=()
for t in 50ns 100ns 200ns; do j=$(sbatch --parsable --dependency=afterok:$jo scripts/one_pass-from_off.sh d $t 0 2 $g x,y,z); ds+=("$j"); echo "d $t -> $j"; done
dep=$(IFS=:; echo "${ds[*]}")
jc=$(sbatch --parsable --dependency=afterok:$dep scripts/concat.sbatch); echo "expt_concat -> $jc"
jl=$(sbatch --parsable --dependency=afterok:$jc scripts/careless-cdef-ohp-mlpw.sh $g); echo "careless    -> $jl"
js=$(sbatch --parsable --dependency=afterok:$jl scripts/stats.sbatch); echo "stats       -> $js"
