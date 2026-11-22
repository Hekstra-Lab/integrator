#!/bin/bash
# Score laue.integrate and every exported arm of one datsimx row against its truth, into one directory.
#
# For each (set, dataset name, laue MTZ) it scores laue.integrate (laue BATCH,
# mapped through --expt) and then each arm's export under
# $EXPORTS/<estimator>/<RUN_PREFIX>_<set>_<arm>/epoch_<EPOCH>/preds_epoch_<EPOCH>.mtz
# (integrator.predict BATCH = image_num + 1), appending to scores.csv,
# scores_by_d.csv and joined.csv in $OUT. Submit from the integrator checkout:
#
#   sbatch -p test -c 2 --mem=24G -t 0-01:00 -o score_%j.log scripts/sim_truth/score_exports.sh
#
# Defaults are the mos010 photon-counting run of 2026-11-22; every path is an
# environment variable.
set -euo pipefail
S=${S:-/n/netscratch/hekstra_lab/Lab/laldama}
ROW=${ROW:-$S/sims/paper_full/mos010}
EXPORTS=${EXPORTS:-$S/sim_runs/mos010_pc_exports}
OUT=${OUT:-$ROW/scores_pc_arms}
RUN_PREFIX=${RUN_PREFIX:-mos010_pc}
EPOCH=${EPOCH:-0059}
ARMS=${ARMS:-"svae svae_bg svae_conjugate"}
ESTIMATORS=${ESTIMATORS:-"rb posterior"}
# set:dataset name:laue MTZ under $ROW/laue_dials_pc
SETS=${SETS:-"default:mos010_pc_default:integrated.mtz all:mos010_pc_permissive:integrated_all.mtz"}
LAUE=${LAUE:-$ROW/laue_dials_pc}
GAIN=${GAIN:-1}
PY=${PY:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/integrator-cuda-dev/bin/python}
SC=${SLURM_SUBMIT_DIR:-$(pwd)}/scripts/sim_truth/score_against_truth.py
T=$ROW/truth/truth_reflections.csv

[ -e "$OUT" ] && { echo "$OUT exists; set OUT to a new directory"; exit 1; }
mkdir -p "$OUT"
common=(--gain "$GAIN" --out "$OUT/scores.csv" --by-d-out "$OUT/scores_by_d.csv" --joined-out "$OUT/joined.csv")
for spec in $SETS; do
  IFS=: read -r set name mtz <<< "$spec"
  $PY -u "$SC" "$T" "$LAUE/$mtz" --method laue_integrate --dataset "$name" \
    --expt "$LAUE/poly_refined.expt" "${common[@]}"
  for arm in $ARMS; do for est in $ESTIMATORS; do
    $PY -u "$SC" "$T" "$EXPORTS/$est/${RUN_PREFIX}_${set}_$arm/epoch_$EPOCH/preds_epoch_$EPOCH.mtz" \
      --method "${arm}_$est" --dataset "$name" --batch-convention image_num "${common[@]}"
  done; done
done
echo SCORES_DONE
