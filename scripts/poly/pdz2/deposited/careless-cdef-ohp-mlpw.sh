#!/bin/bash
#SBATCH --job-name=gpu_careless_integ
#SBATCH -p gpu_requeue # partition (queue)
#SBATCH --mem 32G # memory pool for all cores
#SBATCH -t 0-02:30 # time (D-HH:MM)
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o logs/test_careless_integrate_off_%j.out        # Standard output
#SBATCH -e logs/test_careless_integrate_off_%j.err        # Standard error

##########
#This is an example careless script, hardcoded for analyzing PDZ2 EFX data, and used for running on Harvard's computing cluster. In line 16 and 17 of this script, the relevant conda environment (with careless in it) is activated and relevant modules are loaded so that the script functions on Harvard's computing cluster.  
##########

source /n/lab_storage/hekstra_lab/people/aldama/micromamba/etc/profile.d/mamba.sh
CARELESS="micromamba run -n crls python /n/lab_storage/hekstra_lab/people/aldama/micromamba/envs/crls/bin/careless"

gain=${1}
gain_str=$(echo "$gain" | tr . ,)

##HARDCODED DIRECTORIES
expt=e35
base=gain_${gain_str}_from_stills/
in=${base}/ld_${gain_str}/
out=${base}/careless-cdef-${gain_str}-ohp-mlpw

mkdir $out

BASE_ARGS=(
  --iterations=12_000
  --dmin=1.7 
  --separate-files 
  --test-fraction=0.1
  --positional-encoding-frequencies=5
  --positional-encoding-keys="xcal,ycal"
  --wavelength-key='wavelength' 
  --studentt-likelihood-dof=64
  --merge-half-datasets
  --half-dataset-repeats=3
  --mlp-width=32
  # "xobs,yobs,wavelength,dHKL,BATCH"  # int_test version
  "Hobs,Kobs,Lobs,xcal,ycal,wavelength,dHKL,is_c,is_d,is_e,is_f,BATCH" # integrate version

)

$CARELESS poly \
  ${BASE_ARGS[@]} \
  $in/cdef_${expt}_off.mtz \
  $in/cdef_${expt}_50ns_shared_index.mtz \
  $in/cdef_${expt}_100ns_shared_index.mtz \
  $in/cdef_${expt}_200ns_shared_index.mtz \
  $in/cdef_${expt}_off_P1.mtz \
  $in/cdef_${expt}_50ns_P1.mtz \
  $in/cdef_${expt}_100ns_P1.mtz \
  $in/cdef_${expt}_200ns_P1.mtz \
  $out/merged_${expt}

DURATION=$SECONDS
MESSAGE="Job $SLURM_JOB_ID:careless finished on $HOSTNAME in $(($DURATION / 60)) minutes."
echo $MESSAGE
