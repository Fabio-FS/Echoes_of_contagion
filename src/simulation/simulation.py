import pickle
import os
import numpy as np
import igraph as ig

from .network import initialize_system
from .posts import initialize_posts_arrays
from .disease_dynamics import disease_dynamic_step
from .opinion_dynamic import opinion_dynamic_step_arrays



def simulations(param):
    all_results = []
    for i in range(param["n_of_replicas"]):
        print(f"Running replica {i+1}/{param['n_of_replicas']}")
        RES = single_simulations_arrays(param)
        all_results.append(RES)
    return all_results


def single_simulations_arrays(param):
    """Main simulation with arrays instead of Post objects"""
    g = initialize_system(param)
    post_values, post_upvotes, post_readers = initialize_posts_arrays(g)
    RES = initialize_results(g, param)
    
    for step in range(param["N_steps"]):
        opinion_dynamic_step_arrays(g, post_values, post_upvotes, post_readers, step)
        if step >= param["waiting_time"]:
            disease_dynamic_step(g)
        update_RES(RES, g, step)

    return RES



def initialize_results(g, param):
    n_opinion_samples = param.get("n_opinion_samples", 300)  # Default 200 samples
    save_steps = get_log_sample_steps(param["N_steps"], n_opinion_samples)
    n_saves = len(save_steps)
        # Precompute mapping for faster lookup
    step_to_idx = {step: idx for idx, step in enumerate(save_steps)}
    
    RES = {
        'step_to_idx': step_to_idx,  # Add this
        'save_steps': np.array(save_steps, dtype=np.int32),
        'opinions': np.zeros((n_saves, g["n_humans"]), dtype=np.float32),
        'S_count': np.zeros(param["N_steps"], dtype=np.int16),
        'I_count': np.zeros(param["N_steps"], dtype=np.int16), 
        'R_count': np.zeros(param["N_steps"], dtype=np.int16),
        'mean_opinion': np.zeros(param["N_steps"], dtype=np.float32),
        'opinion_var': np.zeros(param["N_steps"], dtype=np.float32),
        'fraction_ever_infected': np.zeros(param["N_steps"], dtype=np.float32)
    }
    return RES

def update_RES(RES, g, step):
    # Always save aggregates
    health_states = np.array(g.vs['health_state'][:g["n_humans"]], dtype=np.int8)
    RES['S_count'][step] = np.sum(health_states == 0)
    RES['I_count'][step] = np.sum(health_states == 1)
    RES['R_count'][step] = np.sum(health_states == 2)
    
    RES['mean_opinion'][step] = np.mean(g.vs['opinion'][:g["n_humans"]])
    RES['opinion_var'][step] = np.var(g.vs['opinion'][:g["n_humans"]])
    RES['fraction_ever_infected'][step] = (RES['I_count'][step] + RES['R_count'][step]) / g["n_humans"]
    
    # Use dictionary lookup instead of array membership check
    if step in RES['step_to_idx']:
        save_idx = RES['step_to_idx'][step]
        RES['opinions'][save_idx] = np.array(g.vs['opinion'][:g["n_humans"]], dtype=np.float32)

 


def save_results(all_results, param, filename=None):
    """Save results from multiple replicas and parameters to file with optimized data types"""
    if filename is None:
        filename = f"sim_n{param['n_humans']}_bots{param['n_bots']}_reps{param['n_of_replicas']}.pkl"
    
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    # Optimize data types before saving
    optimized_results = []
    for replica in all_results:
        optimized_replica = {}
        for key, value in replica.items():
            if key in ['opinions', 'mean_opinion', 'opinion_var', 'fraction_ever_infected']:
                optimized_replica[key] = value.astype(np.float32)
            elif key == 'health_states':
                optimized_replica[key] = value.astype(np.int8)
            else:
                optimized_replica[key] = value
        optimized_results.append(optimized_replica)
    
    data = {
        'all_results': optimized_results,  # List of optimized replica results
        'parameters': param
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Results from {len(all_results)} replicas saved to {filepath}")

def load_results(filename):
    """Load results from file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['all_results'], data['parameters']


def get_log_sample_steps(N_steps, n_samples=200):
    """Generate exactly n_samples logarithmically spaced steps"""
    if N_steps <= n_samples:
        return list(range(N_steps))
    
    # Logarithmic spacing from 0 to N_steps-1
    # Use log(1 + x) to avoid log(0) and get better early sampling
    log_points = np.logspace(0, np.log10(N_steps), n_samples, dtype=float)
    steps = np.unique(np.round(log_points - 1).astype(int))
    
    # Ensure we have first and last steps
    steps = np.clip(steps, 0, N_steps - 1)
    
    # Remove duplicates and sort
    return sorted(list(set(steps)))