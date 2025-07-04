#!/bin/bash
#SBATCH --job-name=opinion_grid
#SBATCH --array=0-49
#SBATCH --time=02:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Create logs directory
mkdir -p logs

# Load modules (adjust for your cluster)
module load python/3.9

# Run the simulation with the array task ID
python grid_search.py $SLURM_ARRAY_TASK_ID