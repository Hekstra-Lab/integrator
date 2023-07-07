#!/bin/bash
#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-01:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH -o pytorch_%j.out 
#SBATCH -e pytorch_%j.err 


source /n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/etc/profile.d/micromamba.sh


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/envs/

# Load software modules and source conda environment
micromamba activate pt2.0.1_cuda11.8

# Run program
srun -c 1 --gres=gpu:1 python integrate.py
