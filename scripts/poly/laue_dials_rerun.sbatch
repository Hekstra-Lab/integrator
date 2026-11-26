#!/bin/bash
#SBATCH -J ld_rerun
#SBATCH -p sapphire
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -t 1-00:00
#
# Rerun laue.predict and laue.integrate with another laue-dials, from the refined
# geometry an existing dataset was built on, so the classical reference can be
# brought up to date without recutting the integrator's shoeboxes.
#
# hewl_1118's shoeboxes were cut by refltorch.mksbox-laue from laue-dials v0.5.1's
# integrated_HEWL_anom_3049.refl with poly_refined_HEWL_anom_3049.expt, first 1000
# images (pytorch_data/mksbox.sh). laue-dials main (PR #86, 2026-09-18) replaced
# laue.integrate with a profile-fitting integrator that applies Sivia's prior and
# changed laue.predict's masking; indexing and refinement are unchanged, so the
# rerun starts from the same poly_refined files with process.sh's settings.
#
# laue-dials is chosen the way scripts/sim_truth/paper_sims/sim_laue_process.sbatch
# chooses it: LD_ENV (an env prefix; default integrator-cuda-dev, DIALS 3.26) and
# LAUE_DIALS_SRC, a checkout's src/ placed first on PYTHONPATH (default the clean
# main checkout, laue-dials-main at d57f96e). Nothing is installed. See
# docs/installations.md for every checkout and env.
#
# The experiment file names images by their pre-migration path (/n/hekstra_lab/...);
# a copy with the path rewritten (IMAGE_PATH_FROM -> IMAGE_PATH_TO) is written to WORK.
#
#   DATA=/n/lab_storage/hekstra_lab/people/aldama/integrator_data/hewl_1118 WORK=$S/hewl1118_runs/laue_main \
#       sbatch --export=ALL scripts/poly/laue_dials_rerun.sbatch
#
# Afterwards scripts/poly/compare_laue_predictions.py compares the new prediction
# with the reflection list the shoeboxes were cut from and writes the reference MTZ
# restricted to the first MAX_IMAGES images.
set -euo pipefail
MAMBA_ROOT=/n/lab_storage/hekstra_lab/people/aldama/micromamba
E=${LD_ENV:-$MAMBA_ROOT/envs/integrator-cuda-dev}
LAUE_DIALS_SRC=${LAUE_DIALS_SRC:-/n/lab_storage/hekstra_lab/people/aldama/software/laue-dials-main/src}
DATA=${DATA:?set DATA to the dataset directory holding poly_refined_<LD_PREFIX>.expt/.refl}
WORK=${WORK:?set WORK to an output directory}
# LD_PREFIX, not PREFIX: conda/micromamba environments export PREFIX (the env path), and
# inheriting it named the inputs poly_refined_/n/lab_storage/.../local.expt
LD_PREFIX=${LD_PREFIX:-HEWL_anom_3049}
LAM_MIN=${LAM_MIN:-0.97}
LAM_MAX=${LAM_MAX:-1.25}
D_MIN=${D_MIN:-1.4}
MAX_IMAGES=${MAX_IMAGES:-1000}
OURS=${OURS:-$DATA/pytorch_data/reflections_.refl}
IMAGE_PATH_FROM=${IMAGE_PATH_FROM:-/n/hekstra_lab/}
IMAGE_PATH_TO=${IMAGE_PATH_TO:-/n/lab_storage/hekstra_lab/}
NPROC=${NPROC:-${SLURM_CPUS_PER_TASK:-16}}
REPO=${SLURM_SUBMIT_DIR:?submit from a checkout}

export PATH="$E/bin:$PATH"
export PYTHONPATH="$LAUE_DIALS_SRC${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
# laue.predict forks a pool after training a scipy KDE; without these, BLAS locks
# copied into the workers deadlock them (docs/pitfalls.md: 11.5 h -> 92 s)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "===== env $E; laue_dials from $(python -c 'import laue_dials; print(laue_dials.__file__)')" \
     "commit $(git -C "$LAUE_DIALS_SRC" rev-parse --short HEAD 2>/dev/null)"

mkdir -p "$WORK"
cd "$WORK"
sed "s#$IMAGE_PATH_FROM#$IMAGE_PATH_TO#g" "$DATA/poly_refined_$LD_PREFIX.expt" > "poly_refined_$LD_PREFIX.expt"
ln -sf "$DATA/poly_refined_$LD_PREFIX.refl" .
if [ ! -f "predicted_$LD_PREFIX.refl" ]; then
  laue.predict "poly_refined_$LD_PREFIX.expt" "poly_refined_$LD_PREFIX.refl" \
    output.reflections="predicted_$LD_PREFIX.refl" output.log="laue.predict_$LD_PREFIX.log" \
    wavelengths.lam_min=$LAM_MIN wavelengths.lam_max=$LAM_MAX reciprocal_grid.d_min=$D_MIN nproc=$NPROC
fi
if [ ! -f "integrated_$LD_PREFIX.mtz" ]; then
  laue.integrate "poly_refined_$LD_PREFIX.expt" "predicted_$LD_PREFIX.refl" \
    output.filename="integrated_$LD_PREFIX.mtz" output.reflections="integrated_$LD_PREFIX.refl" \
    output.log="laue.integrate_$LD_PREFIX.log" nproc=$NPROC
fi
python "$REPO/scripts/poly/compare_laue_predictions.py" \
  --new-refl "integrated_$LD_PREFIX.refl" --new-mtz "integrated_$LD_PREFIX.mtz" --ours "$OURS" \
  --max-images "$MAX_IMAGES" --out-mtz "integrated_${LD_PREFIX}_first${MAX_IMAGES}.mtz" --out-csv prediction_comparison.csv
echo "LD_RERUN_DONE"
