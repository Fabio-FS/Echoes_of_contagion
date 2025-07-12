#!/bin/bash
#SBATCH --job-name=opinion_grid
#SBATCH --partition=cpu
#SBATCH --array=0-971
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Create logs directory
mkdir -p logs

# Activate conda environment
source /opt/bwhpc/common/devel/miniforge/24.11.0-py3.12/etc/profile.d/conda.sh
conda activate networks

# Navigate to src directory  
cd src

# Run the simulation with the array task ID
python Experiment_4_fixed_sigma_1000_and_30_waiting.py $SLURM_ARRAY_TASK_ID