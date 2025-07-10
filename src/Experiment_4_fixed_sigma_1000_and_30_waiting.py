# grid_search_experiment1.py - Larger Networks
import itertools
import sys
import os
from simulation.simulation import simulations, save_results
import datetime
import glob

# Experiment-specific settings
EXPERIMENT_NAME = "fixed_sigma_1000_Agents"

def generate_parameter_grid():
    """Generate all parameter combinations for larger networks experiment"""
    comm_errors = [0.2]
    thresholds = [-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
    n_bots = [0, 1, 2, 4, 8, 16, 32]
    
    combinations = list(itertools.product(comm_errors, thresholds, n_bots))
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
        "waiting_time": 208,    # 208 time steps = 52 days for the news about the virus to spread before the virus arrives
                                #December 31, 2019: China reported cluster in Wuhan
                                #February 21, 2020: First Italian COVID case diagnosed in Codogno (38-year-old Mattia Maestri)
        "mu": 0.075,
        "epsilon": 0.3,
        "bot_threshold": -0.5,  # Will be overridden by grid
        "beta0": 0.0125,            # R0 = 3, but 6 hours time step now
        "recovery_rate": 0.025,
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
    ce, th, nb = combinations[job_id]
    
    param = get_base_param()
    param["communication_error"] = ce
    param["bot_threshold"] = th
    param["n_bots"] = nb
    
    print(f"Running {EXPERIMENT_NAME} job {job_id}: ce={ce}, th={th}, nb={nb}")
    
    # Run simulation
    consolidated_results = simulations(param)
    
    # Save with unique filename
    filename = f"results_ce{ce}_th{th}_nb{nb}_2025_07_09.pkl"
    filepath = os.path.join(experiment_dir, filename)
    
    save_results(consolidated_results, filepath)
    
    print(f"Job {job_id} completed: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grid_search_experiment1.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    run_single_job(job_id)