#!/bin/bash
#SBATCH -J pdz2_stats
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-02:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model/logs/%x_%j.out
#
# `careless.stats` on the PDZ2 EF-X merges, one arm at a time.
#
# PDZ2 scales outside the pipeline -- its merge is the deposited eight-file
# joint careless call, not config 4 on one file -- so the stats step is
# pointed at each arm's careless output base with `--out-base` rather than
# at a run directory. File index 0 is the field-off C2 shared-index merge,
# the file the refinement endpoint uses.
#
#   sbatch scripts/poly/pdz2/stats_pdz2.sbatch svae
#   sbatch scripts/poly/pdz2/stats_pdz2.sbatch laue w21_k7m geom_both
#
# VENDORED from jobs/pdz2figs_1118/pdz2_stats.sbatch, which lived on
# netscratch and put a staged stub package first on PYTHONPATH because
# "neither cluster checkout carries utils/checkpoints.py or
# utils/metrics.py". Both do now, and that stub is what shadowed the
# installed package twice. It runs from the checkout instead.
set -uo pipefail
M=/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_model
CRL=gain_0,3_from_stills/careless-cdef-0,3-ohp-mlpw/pdz2
REPO=${REPO:-${SLURM_SUBMIT_DIR:?submit from a checkout, or set REPO}}
CFG=${STATS_CFG:-$REPO/scripts/poly/pdz2/pdz2_stats_cfg.yaml}
ENV_NAME=${ENV_NAME:-integrator-cuda-dev}
MAMBA_ROOT=${MAMBA_ROOT:-/n/lab_storage/hekstra_lab/people/aldama/micromamba}
PY=$MAMBA_ROOT/envs/$ENV_NAME/bin/python
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 1; }
[ -s "$CFG" ] || { echo "no stats config at $CFG"; exit 1; }

source "$MAMBA_ROOT/etc/profile.d/mamba.sh"
echo "===== python: $PY"
echo "===== repo:   $REPO ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))"

# `tree_<arm>` for every arm except the reference, which is its own tree
tree_of() {
  case "$1" in
    laue) echo "/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/pdz2_ref" ;;
    k7m_bfix) echo "$M/tree_w21_k7m_bfix" ;;
    *) echo "$M/tree_$1" ;;
  esac
}

# cd away from the checkout: python puts the script's directory first on
# sys.path, and `scripts/poly/pdz2` holds modules named like the package's
cd /tmp || exit 1
for arm in "${@:?name at least one arm}"; do
  base="$(tree_of "$arm")/$CRL"
  if [ ! -s "${base}_0.mtz" ]; then
    echo "SKIP $arm: no ${base}_0.mtz"
    continue
  fi
  echo "===== $arm"
  "$PY" "$REPO/scripts/poly/run_pipeline.py" --cfg "$CFG" \
    --out-base "$base" --method "$arm" --dataset pdz2_1116 2>&1 \
    | grep -viE "^2[0-9]{3}-|tensorflow|oneDNN|cuFFT|cuDNN|cuBLAS|absl::|FutureWarning|grouper\.apply|AVX|rebuild TensorFlow|WARNING: All log"
done
echo "PDZ2_STATS_DONE"
