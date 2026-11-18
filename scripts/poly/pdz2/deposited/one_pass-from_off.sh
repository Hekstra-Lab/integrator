#!/bin/bash
#SBATCH --job-name=PDZ_index_ld
#SBATCH -p shared,serial_requeue # partition (queue)
#SBATCH --mem 64G # memory pool for all cores
#SBATCH -t 0-04:00 # time (D-HH:MM)
#SBATCH -c 4
#SBATCH -o logs/index_%j.out        # Standard output
#SBATCH -e logs/index_%j.err        # Standard error

# An example script for processing an electric-field-stimulated time-resolved dataset. This script can be run using slurm. This script takes six arguments, pass letter (RUN_NO), timepoint (TIME), start phi angle (START_ANGLE), phi angle step (OSC), gain (GAIN), and reindexing operation (CB_OP). See process-slurm.sh for an example script call.

# Local adaptation (processing logic unchanged): laue-dials 0.5.1 env, and pin
# BLAS/OpenMP threads so laue.predict's fork-after-KDE does not deadlock.
source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
micromamba activate integrator-cuda-dev
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

RUN_NO=${1}
TIME=${2}
CELL='"65.2,39.45,38.9,90.000,117.45,90.000"'
GAIN=${5}
CB_OP=${6}
GAIN_STR=$(echo "$GAIN" | tr . ,)
START_ANGLE=${3}
OSC=${4}
N=4 # Max multiprocessing

# Make needed directories
folder_name=gain_${GAIN_STR}_from_stills
mkdir $folder_name
mkdir $folder_name/dials_files_${RUN_NO}_${TIME}
cd $folder_name/dials_files_${RUN_NO}_${TIME}

FILE_INPUT_TEMPLATE=$(pwd)"/../../data/e35${RUN_NO}_${TIME}_###.mccd"

# Import data into DIALS files
dials.import geometry.scan.oscillation=$START_ANGLE,$OSC \
    geometry.goniometer.invert_rotation_axis=True \
    geometry.goniometer.axes=0,1,0 \
    geometry.beam.wavelength=1.04 \
    geometry.detector.panel.pixel_size=0.08854,0.08854 \
    input.template=$FILE_INPUT_TEMPLATE \
    output.experiments=imported.expt

# Get a monochromatic geometry model
laue.find_spots imported.expt \
    spotfinder.mp.nproc=8 \
    spotfinder.threshold.dispersion.gain=$GAIN \
    spotfinder.filter.max_separation=10

laue.index imported.expt strong.refl \
    indexer.indexing.known_symmetry.space_group=5 \
    indexer.indexing.refinement_protocol.mode=refine_shells \
    indexer.indexing.known_symmetry.unit_cell=$CELL \
    indexer.refinement.parameterisation.auto_reduction.action=fix \
    laue_output.index_only=False

#for consistent geometry, some of these data require reindexing. we do this with dials.reindex. 
dials.reindex monochromatic.{expt,refl} change_of_basis_op=$CB_OP

# Split sequence into stills
laue.sequence_to_stills reindexed.expt reindexed.refl

# to maintain spot positions and wavelengths between datasets, given that the crystal position does not change during the 
# timepoints, we copy the geometry and spot positions from the off dataset onto the timepoints. 
# we find and replace the images in the off dataset poly_refined for the current on dataset. 
sed s/e35${RUN_NO}_off_/e35${RUN_NO}_${TIME}_/g ../dials_files_${RUN_NO}_off/stills.expt > stills_rewritten.expt

laue.optimize_indexing stills_rewritten.expt stills.refl \
    output.experiments="optimized.expt" \
    output.reflections="optimized.refl" \
    output.log="laue.optimize_indexing.log" \
    wavelengths.lam_min=0.95 \
    wavelengths.lam_max=1.2 \
    reciprocal_grid.d_min=1.7 \
    nproc=$N

laue.refine optimized.* output.experiments="poly_refined.expt" output.reflections="poly_refined.refl" output.log="laue.poly_refined.log" nproc=$N

laue.predict poly_refined.* \
    output.reflections="predicted.refl" \
    output.log="laue.predict.log" \
    wavelengths.lam_min=0.95 \
    wavelengths.lam_max=1.2 \
    reciprocal_grid.d_min=1.7 \
    nproc=$N

laue.integrate poly_refined.expt predicted.refl output.filename="integrated.mtz" output.log="laue.integrate.log" nproc=$N

laue.compute_rmsds poly_refined.* > rmsds.log


# This is where laue_dials ends. The output file integrated.mtz can be merged in careless and refined in phenix to get a model
