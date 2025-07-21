import itertools
import sys
import os

# Add path to simulation modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from simulation.simulation import simulations, save_results
import datetime

def generate_parameter_grid():
    """Generate all parameter combinations for no-bots sigma experiment"""
    comm_errors = [0.2]
    thresholds = [0]    # Not used in this experiment but kept for consistency
    n_bots = [0]        # No bots experiment
    
    combinations = list(itertools.product(comm_errors, thresholds, n_bots))
    print(f"Total combinations: {len(combinations)}")
    return combinations

def get_base_param():
    """Base parameters for no-bots experiment"""
    return {
        "n_of_replicas": 100,
        "n_humans": 1000,
        "n_bots": 0,         # Will be overridden by grid
        "nei": 6,
        "p": 0.05,
        "N_steps": 5000,
        "waiting_time": 1000,
        "mu": 0.075,
        "epsilon": 0.3,
        "bot_threshold": -0.5,  # Will be overridden by grid
        "beta0": 0.0125*4,
        "recovery_rate": 0.025*4,
        "I0": 2,
        "communication_error": 0.2,  # Will be overridden by grid
        "post_history": 10,
        "feed_size": 5,
        "feed_algorithm": "random",

        "disease_model": "SIRV",        # or "SIR" for original model
        "xi_max": 0.05/4,  # Maximum daily vaccination probability (5% per day)
        "use_discrete_vaccination": True,  # Match susceptibility approach
        "vaccination_groups": 5  # Number of discrete vaccination behavior groups
    }

def run_single_job(job_id):
    """Run simulation for a specific parameter combination"""
    # Create results directory in same folder as script
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
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
    
    print(f"Running experiment_1 job {job_id}: ce={ce}, th={th}, nb={nb}")
    
    # Run simulation
    consolidated_results = simulations(param)
    
    # Save with unique filename
    filename = f"results_ce{ce}_th{th}_nb{nb}_2025_07_09.pkl"
    filepath = os.path.join(results_dir, filename)
    
    save_results(consolidated_results, filepath)
    
    print(f"Job {job_id} completed: {filepath}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Experiment_1_no_bots_several_sigmas.py <job_id>")
        sys.exit(1)
    
    job_id = int(sys.argv[1])
    run_single_job(job_id)