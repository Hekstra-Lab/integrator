#!/bin/bash
# Get the number of files programmatically
NUM_FILES=$(python -c "import json; print(len(json.load(open('parallel_config.json'))['refl_files'])-1)")

# Create the batch script
cat > dials_phenix_job.sh << 'EOT'
#!/bin/bash

# Load any modules needed (if applicable)
# module load python/3.8

# Create logs directory if it doesn't exist
mkdir -p logs

# Print some job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $HOSTNAME"
echo "Started at: $(date)"

# Run the Python script with the task ID as the file index
python process_single.py parallel_config.json $SLURM_ARRAY_TASK_ID

# Print job completion information
echo "Finished at: $(date)"
EOT

# Make the job script executable
chmod +x dials_phenix_job.sh

# Now submit the job with the correct array size
sbatch --job-name=dials_phenix_parallel \
       --output=logs/dials_phenix_%A_%a.out \
       --error=logs/dials_phenix_%A_%a.err \
       --time=12:00:00 \
       --mem=8G \
       --partition=shared\
       --cpus-per-task=1 \
       --array=0-${NUM_FILES} \
       dials_phenix_job.sh

echo "Submitted job array with tasks 0-${NUM_FILES}"
echo "Check status with: squeue -u $USER"
