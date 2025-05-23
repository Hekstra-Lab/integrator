#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=300G
#SBATCH -t 0-10:00
#SBATCH -o write_refls_%j.out
#SBATCH -e write_refls_%j.err



INPUT_PATH=$1

# Check if path was provided
if [ -z "$INPUT_PATH" ]; then
  echo "Error: Path argument is required"
  exit 1
fi

# Load environment

source /n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/etc/profile.d/micromamba.sh

micromamba activate pt2.0.1_cuda11.8

python write_refls.py --path "$INPUT_PATH"
