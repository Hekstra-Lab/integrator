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
M=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model
# the checkout this job was submitted from, so what runs is what is
# committed. SLURM copies a batch script to its spool before running
# it, so $BASH_SOURCE is not the repository.
SRC=${RB_SRC:-${SLURM_SUBMIT_DIR:?submit from a checkout, or set RB_SRC}/src}
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

for arm in "${@:-pdz2_geom_both}"; do
  run=$M/runs/$arm
  ckpt=$(ls -1 $run/files/checkpoints/epoch=*.ckpt 2>/dev/null | sort | tail -1)
  [ -z "$ckpt" ] && { echo "SKIP $arm: no checkpoint"; continue; }
  cfg=$M/rb_cfg_$arm.yaml
  echo "===== $arm <- $(basename $ckpt)"
  micromamba run -n integrator-cuda-dev env PYTHONPATH=$SRC \
    python $SLURM_SUBMIT_DIR/scripts/poly/rb_predict_config.py $run $cfg || { echo "CFG_FAIL $arm"; continue; }
  micromamba run -n integrator-cuda-dev env PYTHONPATH=$SRC \
    python -m integrator.cli.predict -v \
    --config $cfg --ckpt "$ckpt" --out-dir $M/rb_preds/$arm \
    --rb-export --write-mtz 2>&1 | tail -15
done
echo "RB_DONE"
