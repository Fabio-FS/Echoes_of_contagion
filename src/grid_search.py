# grid_search.py
import itertools
import sys
import os
from network_generator import simulations, save_results

def generate_parameter_grid():
    """Generate all parameter combinations"""
    comm_errors = [0, 0.05, 0.1, 0.2]
    thresholds = [-0.8, -0.4, 0, -0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    # Create all combinations
    combinations = list(itertools.product(comm_errors, thresholds, n_bots))
    print(f"Total combinations: {len(combinations)}")
    return combinations

def get_base_param():
    """Your base parameter dictionary"""
    return {
    "n_of_replicas" : 20,
    "n_humans" : 1000,
    "n_bots" : 50,
    "nei" : 6,
    "p" : 0.05,
    "N_steps" : 1000,
    "waiting_time" : 500,
    "mu" : 0.075,
    "epsilon": 0.3,
    "bot_threshold" : -0.5,
    "beta0" : 0.0125*4,
    "recovery_rate" : 0.025*4,
    "I0" : 1,
    "communication_error" : 0.2,
    "post_history": 10,
    "feed_size": 5
}

def run_single_job(job_id):
    """Run simulation for a specific parameter combination"""
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
    
    print(f"Running job {job_id}: ce={ce}, th={th}, nb={nb}")
    
    # Run simulation
    results = simulations(param)
    
    # Save with unique filename
    filename = f"results_ce{ce}_th{th}_nb{nb}_2025_07_04.pkl"
    save_results(results, param, filename)
    
    print(f"Job {job_id} completed: {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grid_search.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    run_single_job(job_id)