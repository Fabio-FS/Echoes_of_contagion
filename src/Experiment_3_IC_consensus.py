# grid_search_experiment1.py - Larger Networks
import itertools
import sys
import os
from simulation.simulation import simulations, save_results
import datetime
import glob

# Experiment-specific settings
EXPERIMENT_NAME = "IC_consensus"

def generate_parameter_grid():
    """Generate all parameter combinations for larger networks experiment"""
    mean     = [-1, -0.9, -0.8, -0.5, 0, 0.5, 0.8, 0.9, 1]
    var     = [0.01, 0.05, 0.1, 0.2, 0.5]

    
    combinations = list(itertools.product(mean, var))
    print(f"Total combinations: {len(combinations)}")
    return combinations

def get_base_param():
    """Base parameters for larger networks"""
    return {
        "n_of_replicas": 100,  # Fewer replicas due to larger size
        "n_humans": 100,      # LARGER NETWORK
        "n_bots": 10,         # Will be overridden by grid
        "nei": 6,             # More neighbors for larger network
        "p": 0.05,
        "N_steps": 10000,      # More steps for larger system
        "waiting_time": 0,
        "mu": 0.075,
        "epsilon": 0.3,
        "bot_threshold": -0.5,  # Will be overridden by grid
        "beta0": 0.0125*4,
        "recovery_rate": 0.025*4,
        "I0": 2,              # More initial infected
        "communication_error": 0.2,  # Will be overridden by grid
        "post_history": 10,
        "feed_size": 5
    }

def read_run_directories():
    """Read the directory paths from the setup file"""
    try:
        with open("current_run_dirs.txt", "r") as f:
            lines = f.read().strip().split("\n")
            pickle_dir = lines[0]
            log_dir = lines[1]
            base_dir = lines[2]
            return pickle_dir, log_dir, base_dir
    except FileNotFoundError:
        return "results", "logs", "."

def run_single_job(job_id):
    """Run simulation for a specific parameter combination"""
    pickle_dir, log_dir, base_dir = read_run_directories()
    
    # Create experiment-specific subdirectory
    experiment_dir = os.path.join(pickle_dir, EXPERIMENT_NAME)
    os.makedirs(experiment_dir, exist_ok=True)
    
    combinations = generate_parameter_grid()
    
    if job_id >= len(combinations):
        print(f"Job ID {job_id} exceeds available combinations ({len(combinations)})")
        return
    
    # Get parameters for this job
    mean, var = combinations[job_id]
    
    param = get_base_param()
    param["opinion_init_type"] = "gaussian"
    param["opinion_mean"] = mean      # M
    param["opinion_std"] = var       # S
    
    print(f"Running {EXPERIMENT_NAME} job {job_id}: mean={mean}, var={var}")
    
    # Run simulation
    consolidated_results = simulations(param)
    
    # Save with unique filename
    today = datetime.datetime.now().strftime("%Y_%m_%d")
    filename = f"results_mean{mean}_var{var}_{today}.pkl"

    filepath = os.path.join(experiment_dir, filename)
    
    save_results(consolidated_results, filepath)
    
    print(f"Job {job_id} completed: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grid_search_experiment1.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    run_single_job(job_id)