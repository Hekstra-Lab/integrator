#!/bin/bash
# Tutorial-faithful PDZ2 EF-X reference run: laue-dials tutorial scripts, adapted only
# for env/partition. Mirrors process-slurm.sh block 1 -> block 2 using SLURM
# dependencies instead of waiting by hand, then expt_concat -> careless -> stats.
# Deviations from the tutorial:
#   - pass d runs the standard one_pass-from_off.sh; the tutorial's pass-d script needs
#     stills built with dials 3.14 / laue-dials 0.3.dev48 (laue-dials issue #49).
#   - block 2 does not re-run `off`: process-slurm.sh would overwrite the reindexed
#     pass-e off geometry while the delay jobs are reading it.
set -euo pipefail
cd /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref
FETCH=${1:?usage: run_pdz2_ref.sh FETCH_JOBID}
declare -A START_ANGLES=( ["c"]=0 ["d"]=92 ["e"]=181 ["f"]=361.5)
declare -A OSCS=( ["c"]=2 ["d"]=2 ["e"]=2 ["f"]=1)
gain=0.3
declare -A OFFJ
all=()
for pass in c d e f; do
  cb=x,y,z; [[ $pass == e ]] && cb=-x,y,-z
  j=$(sbatch --parsable --dependency=afterok:$FETCH scripts/one_pass-from_off.sh $pass off ${START_ANGLES[$pass]} ${OSCS[$pass]} $gain $cb)
  OFFJ[$pass]=$j; all+=("$j"); echo "off   $pass cb=$cb -> $j"
done
for delay in 50ns 100ns 200ns; do
  for pass in c d e f; do
    j=$(sbatch --parsable --dependency=afterok:${OFFJ[$pass]} scripts/one_pass-from_off.sh $pass $delay ${START_ANGLES[$pass]} ${OSCS[$pass]} $gain x,y,z)
    all+=("$j"); echo "$delay $pass -> $j"
  done
done
dep=$(IFS=:; echo "${all[*]}")
jc=$(sbatch --parsable --dependency=afterany:$dep scripts/concat.sbatch); echo "expt_concat -> $jc"
jl=$(sbatch --parsable --dependency=afterok:$jc scripts/careless-cdef-ohp-mlpw.sh $gain); echo "careless    -> $jl"
js=$(sbatch --parsable --dependency=afterok:$jl scripts/stats.sbatch); echo "stats       -> $js"
