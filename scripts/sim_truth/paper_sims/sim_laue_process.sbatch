#!/bin/bash
# laue-dials processing of the 180 nanoBragg sim stills, for the sigma_e comparison
# against the per-pixel truth.
#
# WHY IT DIFFERS FROM scripts/poly/laue_process.sh. That script's steps 1-4
# (dials.import -> laue.find_spots -> laue.index -> laue.sequence_to_stills)
# build a stills experiment list with a crystal per still out of a rotation
# sweep. They cannot run here: dxtbx resolves these CBFs to
# `FormatCBFFullStill`, so `geometry.scan.oscillation` is silently ignored and
# `dials.index` has no sweep. The crystal does not have to be indexed at all,
# because the h5 beside each cbf carries the simulation's exact Umat and Bmat;
# `sim_build_stills.py` attached them and checked the result against the
# ground-truth footprint centroids to a median 1.07 px. `truth.expt` is
# therefore the output of step 4, obtained exactly rather than by fitting.
#
# Both integrators are given this same exact geometry on purpose. The scope is
# INTEGRATION, so the indexing and refinement quality must not vary between the
# arms being compared.
#
# `laue.refine` IS run, because `laue.predict` selects on the
# `used_in_refinement` flag that only refinement sets, and because a refinement
# starting from the truth is the supported path. How far it moves off the truth
# is measured afterwards rather than assumed; if it drifts, the comparison uses
# the unrefined geometry instead.
#
# Resumable: each step runs only if its output is missing.
#
# Overrides (defaults = the hkl_data run as made): OUT (holds truth.expt),
# LD_ENV (an env prefix), ADD_S1 (path of sim_add_s1.py), LAM_MIN, LAM_MAX,
# DMIN, NPROC. For datsimx output: LD_ENV=.../envs/integrator-cuda-dev, which
# reads the NeXus masters (only CBF pixels need `ld`), and
# ADD_S1=scripts/sim_truth/paper_sims/sim_add_s1.py.
#
# Step 0, datsimx only (runs when IMG_FILE is set and truth.expt is missing):
# dials.import of the NeXus master IMG_FILE as stills, with REF_GEOM's beam and
# detector and TRUSTED as the trusted range (datsimx masters declare (0,0)),
# then sim_build_stills.py --datsimx-shots TRUTH_SHOTS attaches each shot's
# true crystal and prints the position check. BUILD_STILLS is its path.
#SBATCH -J sim_laue
#SBATCH -c 16
#SBATCH -p sapphire
#SBATCH --mem=96G
#SBATCH -t 0-16:00
#SBATCH -o /n/netscratch/hekstra_lab/Lab/laldama/sim_laue_%j.log
set -uo pipefail

S=/n/netscratch/hekstra_lab/Lab/laldama
E="${LD_ENV:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/ld}"
OUT="${OUT:-$S/sim/laue180}"
ADD_S1="${ADD_S1:-$S/sim_add_s1.py}"
export PATH="$E/bin:$PATH"
# LAUE_DIALS_SRC (default unset): a laue-dials src/ placed first on PYTHONPATH,
# ahead of the env's editable install. Used for the Eiger runs, whose
# non-square panel the installed predict.py indexes past the mask
# (laue-dials-eiger, branch eiger-row-size). The module path is logged.
if [[ -n "${LAUE_DIALS_SRC:-}" ]]; then
  export PYTHONPATH="$LAUE_DIALS_SRC${PYTHONPATH:+:$PYTHONPATH}"
  echo "laue_dials from: $("$E/bin/python" -c 'import laue_dials; print(laue_dials.__file__)')" \
       "commit $(git -C "$LAUE_DIALS_SRC" rev-parse --short HEAD 2>/dev/null)"
fi

# numpy 2.x removed the binary mode of np.fromstring that dxtbx's
# FormatCBFFull.get_raw_data still calls, so only the `ld` env (numpy 1.26.4)
# can read these CBFs at all.
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
# laue.predict forks a multiprocessing.Pool right after training a scipy KDE.
# fork() copies memory but not threads, so BLAS locks the KDE left held in the
# parent deadlock every worker's first kde.pdf call: 11.5 h -> 92 s.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# the simulated spectrum, measured from the wave_data maps: 1.0200-1.1800 A.
# Deliberately a little wider than that so the predictor is not the thing
# clipping the edges -- the comparison's own cut does that, at [1.0350, 1.1650].
LAM_MIN="${LAM_MIN:-1.00}"
LAM_MAX="${LAM_MAX:-1.20}"
DMIN="${DMIN:-1.15}"          # detector corner reaches ~1.20 A at the short-wavelength end
NPROC="${NPROC:-16}"

have() { [[ -s "$1" ]]; }
IMG_FILE="${IMG_FILE:-}"
if [[ -n "$IMG_FILE" ]]; then mkdir -p "$OUT"; fi
cd "$OUT" || exit 1

if [[ -n "$IMG_FILE" ]]; then
if have "$OUT/truth.expt"; then echo "  [0/5] skip (truth.expt)"; else
echo "===== [0/5] datsimx: import stills, attach the true crystals ====="
dials.import "$IMG_FILE" \
  input.reference_geometry="${REF_GEOM:?REF_GEOM needed with IMG_FILE}" \
  geometry.detector.panel.trusted_range="${TRUSTED:--1000,1000000}" \
  convert_sequences_to_stills=True \
  output.experiments="$OUT/imported.expt" output.log="$OUT/import.log" || exit 1
$E/bin/python "${BUILD_STILLS:?BUILD_STILLS needed with IMG_FILE}" --imported "$OUT/imported.expt" \
  --datsimx-shots "${TRUTH_SHOTS:?TRUTH_SHOTS needed with IMG_FILE}" --out "$OUT/truth.expt" \
  --check-shots 1 90 180 --min-pix 50 || exit 1
fi
fi
echo "===== sim laue-dials: $OUT  ($(laue.version 2>/dev/null | head -1)) ====="
echo "lam=[$LAM_MIN,$LAM_MAX] d_min=$DMIN nproc=$NPROC"

if have "$OUT/strong.refl"; then echo "  [1/5] skip (strong.refl)"; else
echo "===== [1/5] laue.find_spots ====="
# gain 1.0, not the 0.30 used for the real detector: these are simulated
# Poisson counts, so the variance IS the mean.
laue.find_spots "$OUT/truth.expt" \
  output.reflections="$OUT/strong.refl" \
  output.log="$OUT/laue.find_spots.log" \
  spotfinder.mp.nproc="$NPROC" \
  spotfinder.threshold.dispersion.gain=1.0 \
  spotfinder.filter.max_separation=10 || exit 1
fi

if have "$OUT/stills.refl"; then echo "  [2/5] skip (stills.refl)"; else
echo "===== [2/5] add s1 (replaces laue.index + sequence_to_stills) ====="
$E/bin/python "$ADD_S1" "$OUT/truth.expt" "$OUT/strong.refl" \
  --out "$OUT/stills.refl" || exit 1
fi

if have "$OUT/optimized.refl"; then echo "  [3/5] skip (optimized.refl)"; else
echo "===== [3/5] laue.optimize_indexing ====="
laue.optimize_indexing "$OUT/truth.expt" "$OUT/stills.refl" \
  output.experiments="$OUT/optimized.expt" \
  output.reflections="$OUT/optimized.refl" \
  output.log="$OUT/laue.optimize_indexing.log" \
  wavelengths.lam_min="$LAM_MIN" \
  wavelengths.lam_max="$LAM_MAX" \
  reciprocal_grid.d_min="$DMIN" \
  nproc="$NPROC" || exit 1
fi

if have "$OUT/poly_refined.refl"; then echo "  [4/5] skip (poly_refined.refl)"; else
echo "===== [4/5] laue.refine ====="
laue.refine "$OUT/optimized.expt" "$OUT/optimized.refl" \
  output.experiments="$OUT/poly_refined.expt" \
  output.reflections="$OUT/poly_refined.refl" \
  output.log="$OUT/laue.refine.log" \
  nproc="$NPROC" || exit 1
fi

if have "$OUT/predicted.refl"; then echo "  [5a] skip (predicted.refl)"; else
echo "===== [5a] laue.predict ====="
laue.predict "$OUT/poly_refined.expt" "$OUT/poly_refined.refl" \
  output.reflections="$OUT/predicted.refl" \
  output.log="$OUT/laue.predict.log" \
  wavelengths.lam_min="$LAM_MIN" \
  wavelengths.lam_max="$LAM_MAX" \
  reciprocal_grid.d_min="$DMIN" \
  nproc="$NPROC" || exit 1
fi

if have "$OUT/integrated.mtz"; then echo "  [5b] skip (integrated.mtz)"; else
echo "===== [5b] laue.integrate (the baseline arm) ====="
# laue.integrate writes ONLY an unmerged MTZ -- there is no output.reflections
# parameter. The MTZ carries H K L BATCH I SIGI xcal ycal wavelength BG SIGBG,
# with BATCH = id + 1, i.e. the shot number, which is also the BATCH of the truth-mask sum table.
# xcal/ycal are the predicted centroids, so footprints can be shared from it.
# isigi_cutoff is left at its default 2.0: it is the threshold the profile model
# uses to pick reference spots, not a row filter, so changing it would be tuning
# laue.integrate rather than running it.
laue.integrate "$OUT/poly_refined.expt" "$OUT/predicted.refl" \
  output.filename="$OUT/integrated.mtz" \
  output.log="$OUT/laue.integrate.log" \
  nproc="$NPROC" || exit 1
fi

echo "===== done ====="
ls -la "$OUT"/*.refl "$OUT"/*.expt "$OUT"/*.mtz 2>/dev/null
