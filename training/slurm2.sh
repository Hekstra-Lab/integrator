#!/bin/bash
#SBATCH -c 16
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH -t 2-00:00
#SBATCH --mem=64G
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

source /n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/etc/profile.d/micromamba.sh
micromamba activate pt2.0.1_cuda11.8

# Use the OUTPUT_DIR passed from the sbatch command or use a default value if it's not provided
OUTPUT_DIR=${OUTPUT_DIR:-"./train_runs/temp"}
mkdir -p $OUTPUT_DIR

# Create a directory for TensorBoard logs
TB_LOGS_DIR="$OUTPUT_DIR/tb_logs/integrator_model"
mkdir -p $TB_LOGS_DIR

# Capture SLURM job ID
JOB_ID=$SLURM_JOB_ID

# Run program
srun python lightning2.py --epochs 10 --dmodel 32 --batch_size 100 --output_dir $OUTPUT_DIR --learning_rate .001 --p_I_scale .0001 --p_bg_scale .0001 --depth 8 --subset_ratio 1 > $OUTPUT_DIR/pytorch_${JOB_ID}.out 2> $OUTPUT_DIR/pytorch_${JOB_ID}.err
