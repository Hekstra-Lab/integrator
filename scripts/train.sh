#!/bin/bash
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200G
#SBATCH -t 0-10:00
#SBATCH -o pytorch_%j.out
#SBATCH -e pytorch_%j.err

# Check if model type is provided
if [ $# -eq 0 ]; then
    echo "Please provide a model type: FcResNet_Softmax, FcResNet_MVN, CNNResNet_Softmax, CNNResNet_MVN, FcResNet_Dirichlet, or CNNResNet_Dirichlet"
    exit 1
fi

MODEL_TYPE=$1

# Set a more informative job name
if [[ $MODEL_TYPE == FcResNet* ]]; then
    ENCODER="FcResNet"
else
    ENCODER="CNNResNet"
fi

if [[ $MODEL_TYPE == *Softmax ]]; then
    PROFILE="Softmax"
elif [[ $MODEL_TYPE == *MVN ]]; then
    PROFILE="MVN"
else
    PROFILE="Dirichlet"
fi

#SBATCH -J "${ENCODER}_${PROFILE}"

# Load environment
source /n/holylabs/LABS/hekstra_lab/Users/laldama/micromamba/etc/profile.d/micromamba.sh
micromamba activate pt2.0.1_cuda11.8

# Set the path to your project directory
PROJECT_DIR="/n/holylabs/LABS/hekstra_lab/Users/laldama/integratorv2/integrator"

# Change to the project directory
cd $PROJECT_DIR

# Generate configs
echo "Generating configuration files..."
python scripts/generate_configs.py

# Set the path to your config file based on the model type
CONFIG_FILE="$PROJECT_DIR/config/${MODEL_TYPE}_config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE"
    echo "Please check if the model type is correct and the config file exists."
    exit 1
fi

# Set the log directory
LOG_DIR="$PROJECT_DIR/logs/outputs"

# Create the log directory if it doesn't exist
mkdir -p $LOG_DIR

# Backup and fix the experiment counter file
COUNTER_FILE="$PROJECT_DIR/experiment_counter.json"
if [ -f "$COUNTER_FILE" ]; then
    cp "$COUNTER_FILE" "${COUNTER_FILE}.bak_${SLURM_JOB_ID}"
    python -c "
import json
import sys

def fix_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        # Remove any extra closing braces at the end
        while content.endswith('}}'):
            content = content[:-1]

        # Ensure the content ends with a single closing brace
        if not content.endswith('}'):
            content += '}'

        # Try to parse the modified content
        data = json.loads(content)

        # Write the corrected JSON back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f'Successfully fixed experiment counter file: {file_path}')
    except json.JSONDecodeError as e:
        print(f'Error in JSON file {file_path}: {str(e)}')
        print('Please check the backup and fix manually.')
        sys.exit(1)

fix_json_file('$COUNTER_FILE')
"
fi

# Run the training script
echo "Starting training for ${ENCODER} with ${PROFILE} profile, using config: $CONFIG_FILE"
python -u scripts/train.py --config $CONFIG_FILE --log_dir $LOG_DIR 2>&1 | tee -a $LOG_DIR/training_log_${SLURM_JOB_ID}.txt

# Deactivate the micromamba environment
micromamba deactivate

# Move the Slurm output and error files to the log directory
mv pytorch_${SLURM_JOB_ID}.out pytorch_${SLURM_JOB_ID}.err $LOG_DIR/
