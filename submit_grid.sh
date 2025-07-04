#!/bin/bash
#SBATCH --job-name=opinion_grid
#SBATCH --partition=cpu
#SBATCH --array=0-99
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Create logs directory
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate networks

# Navigate to src directory  
cd src

# Run the simulation with the array task ID
python grid_search.py $SLURM_ARRAY_TASK_ID