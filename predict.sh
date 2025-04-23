#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-2:00
#SBATCH -o predict_%j.out
#SBATCH -e predict_%j.err

# Load environment

source /n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/etc/profile.d/micromamba.sh

micromamba activate pt2.0.1_cuda11.8

python predict.py
