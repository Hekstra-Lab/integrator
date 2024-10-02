#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 0-03:00
#SBATCH --mem=100G
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

# Load environment
source /n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/etc/profile.d/micromamba.sh
micromamba activate pt2.0.1_cuda11.8

# Set the path to your project directory
PROJECT_DIR="/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator"

# Set the path to your config file
CONFIG_FILE="$PROJECT_DIR/config/FcResNet_MVNProfile_config.yaml"

# Set the log directory
LOG_DIR="$PROJECT_DIR/logs/outputs"

# Create the log directory if it doesn't exist
mkdir -p $LOG_DIR

# Change to the project directory
cd $PROJECT_DIR

# Run the training script
python scripts/train.py --config $CONFIG_FILE --gpus 1 --log_dir $LOG_DIR

# Deactivate the micromamba environment
micromamba deactivate

# Move the Slurm output and error files to the log directory
mv pytorch_${SLURM_JOB_ID}.out pytorch_${SLURM_JOB_ID}.err $LOG_DIR/
