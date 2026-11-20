#!/bin/bash
# The permissive prediction pass, and laue.integrate on it.
#
# WHY. `laue.predict` keeps a prediction only where a KDE trained on the OBSERVED
# strong spots gives the pair (wavelength, |q|^2) a density of at least 1
# (`cutoff_log_probability = 0`, i.e. log p >= 0). On this simulation spot
# finding returns only ~93 strong spots per shot, because the data are genuinely
# weak -- the truth-mask sum says just 1.7% of footprints reach SNR > 5 -- so the KDE is
# badly undersampled and the cut keeps 392 of the ~3,000 reflections per shot
# that are actually excited.
#
# That subset is not wrong, and it is the supported default, so it is kept as
# the sensitivity domain. But it is 8% of the data, it is chosen by a spectrum
# model rather than by anything about integration, and it leaves our arm with
# 63k training observations against the 3.7M the real runs use. This pass
# predicts everything feasible instead and integrates that, so both arms see the
# same larger set. The permissive set CONTAINS the default one, so the default
# remains available as a subset of a single comparison rather than as a separate
# experiment.
#
# -20 and not -inf: `np.log(probs) >= cutoff` drops probs == 0 either way, since
# -inf is not >= any finite number, and -20 keeps everything down to a density
# of 2e-9.
#SBATCH -J sim_laue_all
#SBATCH -c 16
#SBATCH -p sapphire
#SBATCH --mem=96G
#SBATCH -t 0-12:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/sim_laue_all_%j.log
set -uo pipefail

S=/n/netscratch/hekstra_lab/Lab/laldama
E=/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/ld
OUT=$S/sim/laue180
export PATH="$E/bin:$PATH"
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

LAM_MIN=1.00
LAM_MAX=1.20
DMIN=1.15
NPROC=16

have() { [[ -s "$1" ]]; }
cd "$OUT" || exit 1

if have "$OUT/predicted_all.refl"; then echo "skip predict_all"; else
echo "===== laue.predict, permissive ====="
laue.predict "$OUT/poly_refined.expt" "$OUT/poly_refined.refl" \
  output.reflections="$OUT/predicted_all.refl" \
  output.log="$OUT/laue.predict_all.log" \
  wavelengths.lam_min="$LAM_MIN" \
  wavelengths.lam_max="$LAM_MAX" \
  reciprocal_grid.d_min="$DMIN" \
  cutoff_log_probability=-20 \
  nproc="$NPROC" || exit 1
fi

if have "$OUT/integrated_all.mtz"; then echo "skip integrate_all"; else
echo "===== laue.integrate on the permissive set ====="
laue.integrate "$OUT/poly_refined.expt" "$OUT/predicted_all.refl" \
  output.filename="$OUT/integrated_all.mtz" \
  output.log="$OUT/laue.integrate_all.log" \
  nproc="$NPROC" || exit 1
fi

echo "===== done ====="
ls -la "$OUT"/predicted_all.refl "$OUT"/integrated_all.mtz
