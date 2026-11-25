#!/bin/bash
#SBATCH -c 8
#SBATCH -N 1
#SBATCH -p shared
#SBATCH --mem=64G
#SBATCH -t 0-08:00
#SBATCH -o poly_pipeline_%j.out
#SBATCH -e poly_pipeline_%j.err
#
# Post-training only: predict -> careless -> phenix.refine -> rs.find_peaks.
# The predict step wants a GPU; submit to a GPU partition when including it.
#
# SUBMIT:
#   RUN_DIR=<run-dir> sbatch scripts/poly/pipeline.slurm
#   RUN_DIR=<run-dir> STEPS=careless,phenix,peaks sbatch scripts/poly/pipeline.slurm
#
# PRED_DIR scales a different export of the same run -- the
# Rao-Blackwellized MTZ, say -- without touching the first scaling, because
# careless writes next to the MTZ it is handed:
#
#   RUN_DIR=<run-dir> PRED_DIR=<run-dir>/rb_preds STEPS=careless,stats,phenix,peaks \
#       sbatch scripts/poly/pipeline.slurm
#
# OUT_BASE scores a merge the pipeline did not make -- a reference
# integrator's careless output, say -- naming the base that `_0.mtz` hangs
# off. It replaces RUN_DIR, and METHOD and DATASET then label the rows that
# the stats step writes. PHENIX_DMIN fixes the resolution both refinement
# passes use, which is what makes a reference merged to a different limit
# comparable:
#
#   OUT_BASE=<.../laue_c4/laue> METHOD=laue DATASET=hewl_1118 \
#       STEPS=phenix,peaks PHENIX_DMIN=1.70 sbatch scripts/poly/pipeline.slurm

set -euo pipefail

INTEGRATOR_ROOT="${INTEGRATOR_ROOT:?INTEGRATOR_ROOT must be set (add to ~/.bashrc).}"
OUT_BASE="${OUT_BASE:-}"
# MTZ names an unmerged MTZ to scale, merge, refine and peak-pick directly -- a
# reference integrator's output such as laue.integrate's integrated.mtz. It
# replaces RUN_DIR; careless writes next to it:
#   MTZ=<.../laue_dials_pc/integrated.mtz> STEPS=careless,stats,phenix,peaks \
#       PIPELINE_CFG=<cfg> sbatch -p gpu --gres=gpu:1 scripts/poly/pipeline.slurm
MTZ="${MTZ:-}"
# CARELESS_SEED=<n> merges with that careless seed into <out_dir>_seed<n>, so
# several merging seeds of one MTZ can be averaged (run_pipeline.py --careless-seed).
if [ -n "$MTZ" ]; then
  RUN_DIR=""
elif [ -z "$OUT_BASE" ]; then
  RUN_DIR="${RUN_DIR:?set RUN_DIR to the trained run directory, or OUT_BASE to a merge base.}"
else
  RUN_DIR=""
fi
PIPELINE_CFG="${PIPELINE_CFG:-$INTEGRATOR_ROOT/scripts/poly/poly_pipeline_cfg.yaml}"
STEPS="${STEPS:-predict,careless,phenix,peaks}"
MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
export MAMBA_SH

# `integrator-cuda-dev`, not `integrator`, and the interpreter by absolute
# path: the latter's console scripts carry a pre-migration shebang that no
# longer exists, so anything resolved through one dies with "bad
# interpreter". The same fix as `train.slurm`.
ENV_DRIVER="${ENV_DRIVER:-integrator-cuda-dev}"
MAMBA_ROOT="${MAMBA_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/micromamba}"
source "$MAMBA_SH"
micromamba activate "$ENV_DRIVER"
PY_BIN="$MAMBA_ROOT/envs/$ENV_DRIVER/bin/python"
[ -x "$PY_BIN" ] || { echo "no interpreter at $PY_BIN"; exit 1; }

# The imported package and this checkout are two different git trees here;
# a mismatch is how six architecture arms died on `Unknown integrator name`.
SRC_DIR=$("$PY_BIN" -c 'import integrator, pathlib; print(pathlib.Path(integrator.__file__).parents[2])' 2>/dev/null)
PKG_SHA=$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
REPO_SHA=$(git -C "$INTEGRATOR_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
echo "===== python:  $PY_BIN"
echo "===== source:  $SRC_DIR ($PKG_SHA)"
if [ "$PKG_SHA" != "$REPO_SHA" ]; then
  echo "===== WARNING: checkout is $REPO_SHA, imported package is $PKG_SHA"
fi

"$PY_BIN" "$INTEGRATOR_ROOT/scripts/poly/run_pipeline.py" \
    ${RUN_DIR:+--run-dir "$RUN_DIR"} \
    ${OUT_BASE:+--out-base "$OUT_BASE"} \
    ${MTZ:+--mtz "$MTZ"} \
    --cfg "$PIPELINE_CFG" \
    --steps "$STEPS" \
    ${PRED_DIR:+--pred-dir "$PRED_DIR"} \
    ${METHOD:+--method "$METHOD"} \
    ${DATASET:+--dataset "$DATASET"} \
    ${PHENIX_DMIN:+--phenix-dmin "$PHENIX_DMIN"} \
    ${CARELESS_SEED:+--careless-seed "$CARELESS_SEED"}
