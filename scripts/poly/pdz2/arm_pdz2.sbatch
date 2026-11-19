#!/bin/bash
#SBATCH -J pdz2_arm
#SBATCH -p gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 0-08:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/logs/arm_%x_%j.out
# Take one trained arm through predict -> per-sweep MTZs -> the REFERENCE pipeline
# (expt_concat -> careless -> CCsym -> difference map), so only integration differs
# from laue.integrate.
#
# Reports BOTH endpoints. CCsym is a half-dataset correlation, and laue.integrate's
# KNN profile borrowing feeds both halves, so it is confounded the same way CC-half
# is; difference-density peaks at the Fig 9C/D sites are read from one merged map at
# positions fixed by the deposited model and cannot be inflated that way. On the
# k7m arm the two disagreed in DIRECTION across beta, so neither is reported alone.
set -uo pipefail
ARM=${1:?arm name}
TREE_NAME=${TREE_NAME:-tree_$ARM}
LEGACY_BATCH=${LEGACY_BATCH:-0}
M=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model
REF=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref
CRLS=/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/crls/bin
RUN=$M/runs/pdz2_$ARM
TREE=$M/$TREE_NAME
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
mkdir -p $TREE/scripts $TREE/logs
cp $REF/scripts/{expt_concat.py,custom_ccsym.py,careless-cdef-ohp-mlpw.sh,stats_for_production.sh,wdm.py} $TREE/scripts/ 2>/dev/null
# OUR scripts come from this script's own directory in the repository, not
# from a copy on scratch. `$M/split_pdz2.py` was a copy that predated
# `--estimator`, so this job ran a 22-second predict pass and then died on
# `unrecognized arguments: --estimator model` while the repository had the
# flag all along. The deposited scripts above stay at $REF, where they are
# the originals as run.
# Under sbatch, $BASH_SOURCE is slurmd's spool copy of this file, not the
# file in the repository, so `dirname` of it finds nothing. The submission
# directory is the checkout the job was launched from. PDZ2_SCRIPTS overrides
# both, and the plain `dirname` is the fallback when run outside SLURM.
if [ -n "${PDZ2_SCRIPTS:-}" ]; then
  HERE=$PDZ2_SCRIPTS
elif [ -d "${SLURM_SUBMIT_DIR:-}/scripts/poly/pdz2" ]; then
  HERE=$SLURM_SUBMIT_DIR/scripts/poly/pdz2
else
  HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
for f in split_pdz2.py wdm_peaks_compare.py; do
  [ -s "$HERE/$f" ] || { echo "MISSING $HERE/$f -- run this from the repo checkout"; exit 1; }
  cp "$HERE/$f" $TREE/scripts/
done
ln -sf $REF/5e1y_phases.mtz $TREE/5e1y_phases.mtz

echo "==================== 1) predict ($ARM), LAST checkpoint only"
if [[ ! -s $RUN/run_paths.yaml ]]; then echo "NO_RUN_DIR $RUN"; exit 1; fi
# `integrator.predict --run-dir` with no --ckpt walks EVERY checkpoint, and
# pdz2_w21_k7m has 51. The splitter then takes only the last parquet, so the
# rest is GPU time thrown away. Polychromatic scores the last checkpoint by
# default (scripts/poly/poly_pipeline_cfg.yaml, `epoch: last`) because one
# careless scale is tens of GPU-minutes; the same reasoning applies here.
# Override with SCORE_EPOCH to score a different one.
CKPT=${CKPT:-$(ls -1 $RUN/files/checkpoints/epoch=*.ckpt 2>/dev/null | sort | tail -1)}
if [[ -z "$CKPT" ]]; then echo "NO_CHECKPOINT $RUN"; exit 1; fi
echo "checkpoint: $(basename $CKPT)"
# RB_BACKGROUND_TERM (sample_bg | fixed_bg) sets `integrator.args.
# rb_background_term` for this predict pass only. It acts at the RB export
# and not in training, so one trained run is scored both ways. Each setting
# predicts from a config copy into its own predictions_<term>/, which the
# splitter is pointed at, so the two passes never overwrite each other.
# Unset, the predict pass is the run's own config into predictions/, as
# before. Use a different TREE_NAME per setting.
PRED_ARGS=()
SPLIT_ARGS=()
if [[ -n "${RB_BACKGROUND_TERM:-}" ]]; then
  RB_CFG=$RUN/config_rb_${RB_BACKGROUND_TERM}.yaml
  RB_PRED=$RUN/predictions_${RB_BACKGROUND_TERM}
  micromamba run -n integrator-cuda-dev python - "$RUN/run_paths.yaml" \
      "$RB_CFG" "$RB_BACKGROUND_TERM" <<'PY' || exit 1
import sys, yaml
paths, out, term = sys.argv[1:]
cfg = yaml.safe_load(open(yaml.safe_load(open(paths))["config"]))
cfg.setdefault("integrator", {}).setdefault("args", {})["rb_background_term"] = term
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(f"rb_background_term = {term} -> {out}")
PY
  PRED_ARGS=(--config "$RB_CFG" --out-dir "$RB_PRED")
  SPLIT_ARGS=(--pred-dir "$RB_PRED")
fi
micromamba run -n integrator-cuda-dev integrator.predict \
  --run-dir $RUN --ckpt "$CKPT" ${PRED_ARGS[@]+"${PRED_ARGS[@]}"} 2>&1 | tail -4

echo "==================== 2) split into 16 per-sweep MTZs (BATCH from poly_refined.expt)"
EXTRA=""
[[ "$LEGACY_BATCH" == "1" ]] && EXTRA="--legacy-batch"
# ESTIMATOR picks which intensities are scored: `model` is the posterior and
# `rb` the profile fit computed at export. Both are scored for an arm because
# which one wins is not monotone across arms.
# The splitter runs against the REPOSITORY's `integrator.io`, not the
# environment's. Two different versions are in play on purpose:
#
#   predict  -- the environment's, because that is the code that trained
#               the checkpoint being predicted from, and a model class that
#               has moved since would not be the same model.
#   split    -- the repository's, because `split_pdz2.py` is the
#               repository's and calls `write_mtz_from_preds(intensity_key=,
#               variance_key=)`, which the installed version does not accept:
#                   TypeError: write_mtz_from_preds() got an unexpected
#                   keyword argument 'intensity_key'
#               It is also where the BATCH convention fix lives, and getting
#               BATCH wrong both cuts the wrong frames and corrupts
#               careless's per-image scale model.
#
# The split is a transformation of the prediction parquet with no model in
# it, so pairing it with the repository's IO is sound. Remove this once the
# environment's checkout is the repository's.
REPO_SRC=$(cd "$HERE/../../.." && pwd)/src
[ -d "$REPO_SRC/integrator" ] || { echo "no integrator package at $REPO_SRC"; exit 1; }
micromamba run -n integrator-cuda-dev env PYTHONPATH=$REPO_SRC \
  python $TREE/scripts/split_pdz2.py \
  --run-dir $RUN --tree $TREE --estimator ${ESTIMATOR:-model} $EXTRA \
  ${SPLIT_ARGS[@]+"${SPLIT_ARGS[@]}"} 2>&1 | tail -22
[[ -s $TREE/gain_0,3_from_stills/dials_files_f_off/integrated.mtz ]] || { echo "SPLIT_INCOMPLETE"; exit 1; }

echo "==================== 3) expt_concat (same frame cuts as reference)"
cd $TREE && micromamba run -n integrator-cuda-dev python scripts/expt_concat.py 0.3 2>&1 | tail -4
[[ -s $TREE/gain_0,3_from_stills/ld_0,3/cdef_e35_off_P1.mtz ]] || { echo "CONCAT_INCOMPLETE"; exit 1; }

echo "==================== 4) careless (tutorial flags, 8 files)"
cd $TREE && bash scripts/careless-cdef-ohp-mlpw.sh 0.3 2>&1 | grep -viE "WARNING|tensorflow|deprecat" | tail -5

echo "==================== 5) CCsym + CCpred"
cd $TREE && bash scripts/stats_for_production.sh 0.3 2>&1 | tail -3
echo "-------------------- CCsym ($ARM)"
ARM=$ARM TREE_NAME=$TREE_NAME $CRLS/python - <<'PY'
import collections, csv, os
tree = os.environ["TREE_NAME"]
p = (f"/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/{tree}"
     "/gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw/e35_ccsym.csv")
rows = list(csv.DictReader(open(p)))
agg = collections.defaultdict(list)
bb = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    try: v = float(r["CCsym"])
    except ValueError: continue
    agg[r["delay"]].append(v); bb[r["delay"]][int(r["bin"])].append(v)
for d in ("off", "50ns", "100ns", "200ns"):
    if d in agg:
        per = " ".join("%+.3f" % (sum(v)/len(v)) for _, v in sorted(bb[d].items()))
        print("CCsym %-6s %-6s mean %+.3f | %s" % (os.environ["ARM"], d, sum(agg[d])/len(agg[d]), per))
PY

echo "==================== 6) weighted difference map + Fig 9C/D peaks"
cd $TREE && micromamba run -n integrator-cuda-dev python scripts/wdm.py 0.3 2>&1 | tail -3
if [[ -s $TREE/gain_0,3_from_stills/wdm_rescut.mtz ]]; then
  micromamba run -n integrator-cuda-dev python $TREE/scripts/wdm_peaks_compare.py \
    --arm "laue=$REF/gain_0,3_from_stills/wdm_rescut.mtz" \
    --arm "$ARM=$TREE/gain_0,3_from_stills/wdm_rescut.mtz" 2>&1 | tail -30
else
  echo "WDM_MISSING"
fi
echo "ARM_DONE_$ARM"
