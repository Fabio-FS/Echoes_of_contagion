#!/bin/bash
#SBATCH --job-name=opinion_grid
#SBATCH --partition=cpu
#SBATCH --array=0-135
#SBATCH --time=12:00:00
#SBATCH --mem=16G
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
python Experiment_7_SIR_random.py $SLURM_ARRAY_TASK_ID