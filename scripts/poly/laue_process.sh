#!/bin/bash
# Tutorial-faithful laue-dials processing for one Laue sweep.
#
# Reproduces the command sequence from the rs-station laue-dials tutorials
# (PDZ2 combine-passes / HEWL anomalous):
#
#   dials.import -> laue.find_spots -> laue.index -> laue.sequence_to_stills
#   -> laue.optimize_indexing -> laue.refine -> laue.predict -> laue.integrate
#
# It produces both the input our integrator consumes (predicted.refl + stills)
# and the laue-dials baseline we compare against (integrated.mtz). All crystal-
# and geometry-specific values are parameters, so the same script runs any
# single sweep; PDZ2's multi-pass combine is this script per pass plus a merge.
#
# Tutorial: https://rs-station.github.io/laue-dials/tutorials/tutorials.html
#
# Usage (defaults are for DHFR / SBGrid 1117):
#   DATA_DIR=... OUT=... ./laue_process.sh
# Override any parameter via the environment, e.g. SG=5 CELL=... for PDZ2.
set -euo pipefail

# ---- environment ----
MAMBA_SH="${MAMBA_SH:-/n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh}"
ENV_LD="${ENV_LD:-integrator-cuda-dev}"     # laue-dials env (the `ld` env's shebangs are broken by the /n/hekstra_lab migration)

# ---- inputs / outputs ----
DATA_DIR="${DATA_DIR:-/n/lab_storage/hekstra_lab/people/aldama/integrator_data/1117}"
IMG_TEMPLATE="${IMG_TEMPLATE:-$DATA_DIR/images/e080_###.mccd}"
OUT="${OUT:-/n/netscratch/hekstra_lab/Lab/laldama/laue_runs/1117}"
TAG="${TAG:-dhfr}"                          # output basename tag (no misleading _1118)

# ---- geometry (dials.import) ----
OSC="${OSC:-0,1}"                           # scan.oscillation = start,step (deg)
GONIO_AXES="${GONIO_AXES:-0,1,0}"
BEAM_WL="${BEAM_WL:-1.03}"                   # central wavelength (A)
IMAGE_RANGE="${IMAGE_RANGE:-1,180}"
INVERT_ROT="${INVERT_ROT:-false}"           # tutorial sets True for PDZ2; DHFR indexes without it

# ---- crystal / spectrum ----
SG="${SG:-19}"                              # space group number (DHFR P2_1 2_1 2_1)
CELL="${CELL-31.773,42.656,93.465,90,90,90}"   # known unit cell; CELL= (empty) omits it, unset = this default
LAM_MIN="${LAM_MIN:-0.9}"
LAM_MAX="${LAM_MAX:-1.2}"
DMIN="${DMIN:-1.4}"
GAIN="${GAIN:-0.30}"
PIXEL="${PIXEL:-}"                          # detector pixel size mm "x,y"; empty = read from header
INDEX_METHOD="${INDEX_METHOD:-}"            # laue.index method; real_space_grid_search (needs CELL) holds the
                                            # known cell over many frames where fft3d shrinks it; empty = fft3d default
NPROC="${NPROC:-8}"

source "$MAMBA_SH"
micromamba activate "$ENV_LD"

# laue.predict forks a multiprocessing.Pool right after training a scipy KDE
# (Pool.map(kde.pdf, ...)). fork() copies memory but not threads, so the BLAS
# locks the KDE left held in the parent deadlock every worker's first kde.pdf
# call. Pinning BLAS/OpenMP to one thread removes the thread pool that the fork
# corrupts (and avoids nproc x BLAS oversubscription in the parallel steps).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT"
cd "$OUT"
echo "===== laue-dials processing: $TAG  ($(laue.version 2>/dev/null | head -1 || echo laue-dials)) ====="
echo "DATA_DIR=$DATA_DIR  OUT=$OUT  SG=$SG  CELL=$CELL  lam=[$LAM_MIN,$LAM_MAX]  d_min=$DMIN"

# Resumable: a step runs only if its final output is missing, so a walltime
# timeout (or a maintenance-window cutoff) resumes from where it stopped
# instead of redoing the slow predict.
have() { [[ -s "$1" ]]; }

if have "$OUT/imported.expt"; then echo "  [1/8] skip (imported.expt exists)"; else
echo "===== [1/8] dials.import ====="
inv=(); [[ "$INVERT_ROT" == "true" ]] && inv=(geometry.goniometer.invert_rotation_axis=True)
pix=(); [[ -n "$PIXEL" ]] && pix=(geometry.detector.panel.pixel="$PIXEL")
dials.import \
  geometry.scan.oscillation="$OSC" \
  geometry.goniometer.axes="$GONIO_AXES" \
  "${inv[@]}" \
  "${pix[@]}" \
  geometry.beam.wavelength="$BEAM_WL" \
  geometry.scan.image_range="$IMAGE_RANGE" \
  input.template="$IMG_TEMPLATE" \
  output.experiments="$OUT/imported.expt" \
  output.log="$OUT/dials.import.log"
fi

if have "$OUT/strong.refl"; then echo "  [2/8] skip (strong.refl exists)"; else
echo "===== [2/8] laue.find_spots ====="
laue.find_spots "$OUT/imported.expt" \
  output.reflections="$OUT/strong.refl" \
  output.log="$OUT/laue.find_spots.log" \
  spotfinder.mp.nproc="$NPROC" \
  spotfinder.threshold.dispersion.gain="$GAIN" \
  spotfinder.filter.max_separation=10
fi

if have "$OUT/monochromatic.refl"; then echo "  [3/8] skip (monochromatic.refl exists)"; else
echo "===== [3/8] laue.index ====="
# A fixed unit cell is optional: passing only the space group lets laue.index
# refine the cell, which is more robust when the fft3d solution is a sublattice
# (HEWL). Set CELL to constrain it (DHFR), leave empty to omit.
cellarg=(); [[ -n "$CELL" ]] && cellarg=(indexer.indexing.known_symmetry.unit_cell="$CELL")
methodarg=(); [[ -n "$INDEX_METHOD" ]] && methodarg=(indexer.indexing.method="$INDEX_METHOD")
laue.index "$OUT/imported.expt" "$OUT/strong.refl" \
  laue_output.final_output.experiments="$OUT/monochromatic.expt" \
  laue_output.final_output.reflections="$OUT/monochromatic.refl" \
  laue_output.log="$OUT/laue.index.log" \
  indexer.indexing.nproc="$NPROC" \
  indexer.indexing.known_symmetry.space_group="$SG" \
  "${cellarg[@]}" \
  "${methodarg[@]}" \
  indexer.indexing.refinement_protocol.mode=refine_shells \
  indexer.refinement.parameterisation.auto_reduction.action=fix \
  laue_output.index_only=False
fi

if have "$OUT/stills.refl"; then echo "  [4/8] skip (stills.refl exists)"; else
echo "===== [4/8] laue.sequence_to_stills ====="
laue.sequence_to_stills "$OUT/monochromatic.expt" "$OUT/monochromatic.refl" \
  output.experiments="$OUT/stills.expt" \
  output.reflections="$OUT/stills.refl" \
  output.log="$OUT/laue.sequence_to_stills.log"
fi

if have "$OUT/optimized.refl"; then echo "  [5/8] skip (optimized.refl exists)"; else
echo "===== [5/8] laue.optimize_indexing ====="
laue.optimize_indexing "$OUT/stills.expt" "$OUT/stills.refl" \
  output.experiments="$OUT/optimized.expt" \
  output.reflections="$OUT/optimized.refl" \
  output.log="$OUT/laue.optimize_indexing.log" \
  wavelengths.lam_min="$LAM_MIN" \
  wavelengths.lam_max="$LAM_MAX" \
  reciprocal_grid.d_min="$DMIN" \
  nproc="$NPROC"
fi

if have "$OUT/poly_refined.refl"; then echo "  [6/8] skip (poly_refined.refl exists)"; else
echo "===== [6/8] laue.refine ====="
laue.refine "$OUT/optimized.expt" "$OUT/optimized.refl" \
  output.experiments="$OUT/poly_refined.expt" \
  output.reflections="$OUT/poly_refined.refl" \
  output.log="$OUT/laue.refine.log" \
  nproc="$NPROC"
fi

if have "$OUT/predicted.refl"; then echo "  [7/8] skip (predicted.refl exists)"; else
echo "===== [7/8] laue.predict ====="
laue.predict "$OUT/poly_refined.expt" "$OUT/poly_refined.refl" \
  output.reflections="$OUT/predicted.refl" \
  output.log="$OUT/laue.predict.log" \
  wavelengths.lam_min="$LAM_MIN" \
  wavelengths.lam_max="$LAM_MAX" \
  reciprocal_grid.d_min="$DMIN" \
  nproc="$NPROC"
fi

if have "$OUT/integrated.mtz"; then echo "  [8/8] skip (integrated.mtz exists)"; else
echo "===== [8/8] laue.integrate (baseline) ====="
laue.integrate "$OUT/poly_refined.expt" "$OUT/predicted.refl" \
  output.filename="$OUT/integrated.mtz" \
  output.reflections="$OUT/integrated.refl" \
  output.log="$OUT/laue.integrate.log" \
  nproc="$NPROC"
fi

echo "===== done: $OUT ====="
echo "  baseline MTZ:   $OUT/integrated.mtz"
echo "  for mksbox:     $OUT/predicted.refl  (+ stills.expt, images)"
