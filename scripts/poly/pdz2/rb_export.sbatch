#!/bin/bash
#SBATCH -J pdz2_rb
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 0-04:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/logs/%x_%j.out
#
# VENDORED from laue_runs/pdz2_model/pdz2_rb.sbatch, which lived on
# netscratch and so was one purge away from gone. `gpu_requeue` had 15
# nodes down and 6 draining and left this pending for over three hours,
# so the partition is now `gpu`. SBATCH directives are not shell-expanded,
# so there is no variable here: override it on the command line, which
# takes precedence, with `sbatch -p <other> ...`.
# Arms come from the command line:
#
#   sbatch scripts/poly/pdz2/rb_export.sbatch pdz2_geom_both pdz2_w21_k7m
# Test the Rao-Blackwellized export on PDZ2. No retraining.
#
# PDZ2's headline result -- prior-free ML beats laue.integrate on R-free by
# 0.0056 -- came from ml_from_preds.py, a one-off script. `i_rb`/`var_rb` is
# the maintained version of the same estimator, with a better sigma: it fits
# once per profile sample and adds the spread by the law of total variance,
# and its conditional term is the Schur complement of the (I,b) information
# rather than 1/F_II.
#
# Two things this job had to learn the hard way, both recorded here so the
# next person does not repeat them:
#   1. The console script resolves to the lab_storage checkout, which
#      predates --rb-export. The current source is staged in the job
#      directory and put first on the path.
#   2. --rb-export only tells the MTZ writer which columns to READ. The
#      parquet's columns come from predict_keys, so the config has to ask
#      for i_rb and var_rb or the export dies on a KeyError at the end,
#      after the whole predict pass has run.
set -uo pipefail
# Defaults are the PDZ2 layout this was written for, so an existing call is
# unchanged. RB_ROOT, RB_RUNS and RB_OUT point it at another dataset --
# hewl_1118's arms live at <root>/<arm> rather than <root>/runs/<arm> --
# and RB_OUT names the export directory, so a second export with a
# different estimator setting does not overwrite the first.
M=${RB_ROOT:-/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model}
RUNS=${RB_RUNS:-$M/runs}
OUT=${RB_OUT:-$M/rb_preds}
# EXPORT=posterior writes the posterior mean and variance (qi_mean/qi_var)
# into the MTZ instead of i_rb/var_rb, from the same checkpoint, so both
# estimators of one run can be scored. Default rb, the existing behaviour.
EXPORT=${EXPORT:-rb}
# CKPT_EPOCH=<n> exports the checkpoint of that epoch (files/checkpoints/epoch=<n>.ckpt)
# instead of the last one, to score a training trajectory. Unset: the last, as before.
# Give each epoch its own RB_OUT: the per-run test_preds_all.parquet is rewritten per export.
CKPT_EPOCH=${CKPT_EPOCH:-}
case $EXPORT in
  rb) EXPORT_FLAG=--rb-export ;;
  posterior) EXPORT_FLAG= ;;
  lc) EXPORT_FLAG=--lc-export ;;
  *) echo "EXPORT must be rb, posterior or lc, got $EXPORT"; exit 1 ;;
esac
# the derived config goes next to the ROOT, which exists; writing it under
# $OUT fails before predict has had a chance to create $OUT
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 1; }
# the checkout this job was submitted from, so what runs is what is
# committed. SLURM copies a batch script to its spool before running
# it, so $BASH_SOURCE is not the repository.
SRC=${RB_SRC:-${SLURM_SUBMIT_DIR:?submit from a checkout, or set RB_SRC}/src}
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

for arm in "${@:-pdz2_geom_both}"; do
  run=$RUNS/$arm
  if [ -n "$CKPT_EPOCH" ]; then
    ckpt=$run/files/checkpoints/epoch=$(printf %04d "$((10#$CKPT_EPOCH))").ckpt
    [ -f "$ckpt" ] || ckpt=
  else
    ckpt=$(ls -1 $run/files/checkpoints/epoch=*.ckpt 2>/dev/null | sort | tail -1)
  fi
  [ -z "$ckpt" ] && { echo "SKIP $arm: no checkpoint${CKPT_EPOCH:+ at epoch $CKPT_EPOCH}"; continue; }
  cfg=$M/${EXPORT}_cfg_$arm.yaml
  echo "===== $arm <- $(basename $ckpt)"
  micromamba run -n integrator-cuda-dev env PYTHONPATH=$SRC \
    python $SLURM_SUBMIT_DIR/scripts/poly/rb_predict_config.py $run $cfg || { echo "CFG_FAIL $arm"; continue; }
  micromamba run -n integrator-cuda-dev env PYTHONPATH=$SRC \
    python -m integrator.cli.predict -v \
    --config $cfg --ckpt "$ckpt" --out-dir $OUT/$arm \
    $EXPORT_FLAG --write-mtz 2>&1 | tail -15
done
echo "RB_DONE"
