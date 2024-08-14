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

# Capture the training script as the first argument
TRAIN_SCRIPT="$1"

# Infer the model type by extracting the name before the ".py" extension
MODEL_TYPE=$(basename "$TRAIN_SCRIPT" .py | cut -d'_' -f2)

# Generate the base directory name using time and job ID
UNIQUE_TIME=$(date +%Y%m%d)

# Initialize the counter
COUNTER=1

# Construct the directory path with counter
OUT_DIR="out/out_${MODEL_TYPE}/out-${UNIQUE_TIME}-run${COUNTER}"

# Loop to find the next available directory name
while [ -d "$OUT_DIR" ]; do
  COUNTER=$((COUNTER + 1))
  OUT_DIR="out/out_${MODEL_TYPE}/out-${UNIQUE_TIME}-run${COUNTER}"
done

# Create the output directory
mkdir -p $OUT_DIR

# Shift the positional parameters to remove the training script argument
shift

# Run the training script
srun python "$TRAIN_SCRIPT" --out_dir $OUT_DIR "$@"

# After the training script completes, run the plot script
srun python plot.py --refl_file "$OUT_DIR/out.refl" \
                    --metadata_pkl_file "$OUT_DIR/results.pkl" \
                    --modeltype $MODEL_TYPE \
                    --out_dir $OUT_DIR

# Move SLURM output files to the output directory
mv pytorch_${SLURM_JOB_ID}.out $OUT_DIR/
mv pytorch_${SLURM_JOB_ID}.err $OUT_DIR/
