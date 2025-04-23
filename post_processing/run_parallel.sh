#!/bin/bash
# Set up error handling
set -e

echo "===== Starting DIALS-Phenix Parallel Processing Setup ====="

# Step 1: Create logs directory
echo "Creating logs directory..."
mkdir -p logs

# Step 2: Generate the configuration file
# give config file path
echo "Generating configuration file..."
python create_config.py --path "$1"

# Step 3: Submit the jobs
echo "Submitting Slurm job array..."
bash submit_jobs.sh

echo "===== Setup Complete ====="
echo "DIALS/PHENIX are now running in parallel."
