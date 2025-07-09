import pickle
import os
import numpy as np
import igraph as ig

from .network import initialize_system
from .posts import initialize_posts_arrays
from .disease_dynamics import disease_dynamic_step
from .opinion_dynamic import opinion_dynamic_step_arrays


def simulations(param):
    """Run multiple simulation replicas and return consolidated results"""
    all_results = []
    for i in range(param["n_of_replicas"]):
        print(f"Running replica {i+1}/{param['n_of_replicas']}")
        # Pass replica index to determine if we save trajectories
        save_trajectories = i < 5  # Only first 5 replicas
        RES = single_simulations_arrays(param, save_trajectories=save_trajectories)
        all_results.append(RES)
    
    # Consolidate results for efficient analysis
    consolidated = consolidate_results(all_results, param)
    return consolidated


def single_simulations_arrays(param, save_trajectories=True):
    """Main simulation with optional trajectory saving"""
    g = initialize_system(param)
    post_values, post_upvotes, post_readers = initialize_posts_arrays(g)
    RES = initialize_results(g, param, save_trajectories=save_trajectories)
    
    for step in range(param["N_steps"]):
        if step % 500 == 0:
            print(f"Step {step} of {param['N_steps']}")
        opinion_dynamic_step_arrays(g, post_values, post_upvotes, post_readers, step)
        if step >= param["waiting_time"]:
            disease_dynamic_step(g)
        update_RES(RES, g, step, save_trajectories=save_trajectories)

    return RES


def initialize_results(g, param, save_trajectories=True):
    """Initialize results with optional trajectory arrays"""
    n_opinion_samples = param.get("n_opinion_samples", 300)  # Default 300 samples
    save_steps = get_log_sample_steps(param["N_steps"], n_opinion_samples)
    n_saves = len(save_steps)
    # Precompute mapping for faster lookup
    step_to_idx = {step: idx for idx, step in enumerate(save_steps)}
    
    RES = {
        'step_to_idx': step_to_idx,
        'save_steps': np.array(save_steps, dtype=np.int32),
        'S_count': np.zeros(param["N_steps"], dtype=np.int16),
        'I_count': np.zeros(param["N_steps"], dtype=np.int16), 
        'R_count': np.zeros(param["N_steps"], dtype=np.int16),
        'mean_opinion': np.zeros(param["N_steps"], dtype=np.float32),
        'opinion_var': np.zeros(param["N_steps"], dtype=np.float32),
        'fraction_ever_infected': np.zeros(param["N_steps"], dtype=np.float32),
        'opinion_bins': np.zeros((param["N_steps"], 20), dtype=np.float32),  # 20 bins
        'has_trajectories': save_trajectories  # Flag for analysis
    }
    
    # Only create opinions array for trajectory-saving replicas
    if save_trajectories:
        RES['opinions'] = np.zeros((n_saves, g["n_humans"]), dtype=np.float32)
    
    return RES


def update_RES(RES, g, step, save_trajectories=True):
    """Update results with optional trajectory saving"""
    # Always save aggregates
    health_states = np.array(g.vs['health_state'][:g["n_humans"]], dtype=np.int8)
    RES['S_count'][step] = np.sum(health_states == 0)
    RES['I_count'][step] = np.sum(health_states == 1)
    RES['R_count'][step] = np.sum(health_states == 2)
    
    opinions = np.array(g.vs['opinion'][:g["n_humans"]])
    RES['mean_opinion'][step] = np.mean(opinions)
    RES['opinion_var'][step] = np.var(opinions)
    RES['fraction_ever_infected'][step] = (RES['I_count'][step] + RES['R_count'][step]) / g["n_humans"]
    
    # Calculate opinion distribution bins
    RES['opinion_bins'][step] = calculate_opinion_bins(opinions)
    
    # Only save individual opinions for trajectory replicas
    if save_trajectories and step in RES['step_to_idx']:
        save_idx = RES['step_to_idx'][step]
        RES['opinions'][save_idx] = opinions.astype(np.float32)


def consolidate_results(all_results, param):
    """Convert list of dictionaries to structured arrays for efficient analysis"""
    n_replicas = len(all_results)
    n_steps = param["N_steps"]
    
    # Find trajectory replicas
    trajectory_replicas = [i for i, res in enumerate(all_results) if res.get('has_trajectories', False)]
    n_trajectory = len(trajectory_replicas)
    
    # Get dimensions from first replica
    first_replica = all_results[0]
    if 'opinions' in first_replica:
        n_saves, n_humans = first_replica['opinions'].shape
    else:
        # Get from trajectory replica
        traj_replica = all_results[trajectory_replicas[0]] if trajectory_replicas else None
        if traj_replica and 'opinions' in traj_replica:
            n_saves, n_humans = traj_replica['opinions'].shape
        else:
            n_saves, n_humans = 0, param['n_humans']
    
    consolidated = {
        'parameters': param,
        'n_replicas': n_replicas,
        'trajectory_replica_indices': np.array(trajectory_replicas),
        
        # Aggregate data - all replicas (n_replicas, n_steps)
        'S_count': np.zeros((n_replicas, n_steps), dtype=np.int16),
        'I_count': np.zeros((n_replicas, n_steps), dtype=np.int16),
        'R_count': np.zeros((n_replicas, n_steps), dtype=np.int16),
        'mean_opinion': np.zeros((n_replicas, n_steps), dtype=np.float32),
        'opinion_var': np.zeros((n_replicas, n_steps), dtype=np.float32),
        'fraction_ever_infected': np.zeros((n_replicas, n_steps), dtype=np.float32),
        'opinion_bins': np.zeros((n_replicas, n_steps, 20), dtype=np.float32),  # 20 bins
        
        # Trajectory data - only subset of replicas (n_trajectory, n_saves, n_humans)
        'opinions': np.zeros((n_trajectory, n_saves, n_humans), dtype=np.float32) if n_trajectory > 0 else None,
        'save_steps': first_replica['save_steps'] if 'save_steps' in first_replica else None,
        
        # Bin definitions for reference
        'bin_edges': np.arange(-1.0, 1.1, 0.1),
        'bin_labels': [f'({edge:.1f},{edge+0.1:.1f})' for edge in np.arange(-1.0, 1.0, 0.1)]
    }
    
    # Fill aggregate arrays
    for i, result in enumerate(all_results):
        consolidated['S_count'][i] = result['S_count']
        consolidated['I_count'][i] = result['I_count'] 
        consolidated['R_count'][i] = result['R_count']
        consolidated['mean_opinion'][i] = result['mean_opinion']
        consolidated['opinion_var'][i] = result['opinion_var']
        consolidated['fraction_ever_infected'][i] = result['fraction_ever_infected']
        consolidated['opinion_bins'][i] = result['opinion_bins']
    
    # Fill trajectory arrays
    if n_trajectory > 0:
        traj_idx = 0
        for i, result in enumerate(all_results):
            if result.get('has_trajectories', False):
                consolidated['opinions'][traj_idx] = result['opinions']
                traj_idx += 1
    
    return consolidated


def analyze_consolidated_results(consolidated):
    """Example analysis functions using consolidated structure"""
    
    # Easy statistics across replicas
    mean_final_opinion = np.mean(consolidated['mean_opinion'][:, -1])
    std_final_opinion = np.std(consolidated['mean_opinion'][:, -1])
    
    # Vectorized operations
    polarization = np.mean(consolidated['opinion_var'], axis=1)  # Average variance per replica
    
    # Opinion distribution analysis
    final_bins = consolidated['opinion_bins'][:, -1, :]  # Final distribution for all replicas
    mean_final_distribution = np.mean(final_bins, axis=0)
    
    # Extreme opinion fractions (< -0.8 and > 0.8)
    extreme_negative = np.mean(consolidated['opinion_bins'][:, :, 0], axis=0)  # Time series
    extreme_positive = np.mean(consolidated['opinion_bins'][:, :, -1], axis=0)  # Time series
    
    # Time series analysis
    convergence_time = []
    for replica in range(consolidated['n_replicas']):
        opinion_ts = consolidated['mean_opinion'][replica]
        # Find when opinion stabilizes (example)
        diff = np.abs(np.diff(opinion_ts[-1000:]))  # Last 1000 steps
        conv_time = np.where(diff < 0.01)[0]
        convergence_time.append(conv_time[0] if len(conv_time) > 0 else -1)
    
    return {
        'mean_final_opinion': mean_final_opinion,
        'std_final_opinion': std_final_opinion,
        'polarization_per_replica': polarization,
        'convergence_times': np.array(convergence_time),
        'final_distribution': mean_final_distribution,
        'extreme_negative_timeseries': extreme_negative,
        'extreme_positive_timeseries': extreme_positive,
        'bin_labels': consolidated['bin_labels']
    }


def save_results(consolidated, filename=None):
    """Save consolidated results"""
    if filename is None:
        param = consolidated['parameters']
        filename = f"consolidated_n{param['n_humans']}_bots{param['n_bots']}_reps{consolidated['n_replicas']}.pkl"
    
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(consolidated, f)
    
    print(f"Consolidated results saved to {filepath}")
    return filepath


def load_results(filename):
    """Load consolidated results"""
    with open(filename, 'rb') as f:
        return pickle.load(f)


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


def calculate_opinion_bins(opinions):
    """Calculate fraction of agents in each opinion bin"""
    # Define bin edges: 20 bins from -1.0 to 1.0 with 0.1 width each
    bin_edges = np.arange(-1.0, 1.1, 0.1)
    
    # Use np.histogram to count agents in each bin
    counts, _ = np.histogram(opinions, bins=bin_edges)
    
    # Convert to fractions
    fractions = counts / len(opinions)
    
    return fractions.astype(np.float32)


def run_and_save_simulation(param):
    """Complete workflow: run simulation, consolidate, and save"""
    
    # Run simulation (returns consolidated results)
    consolidated = simulations(param)
    
    # Save consolidated version
    filepath = save_results(consolidated)
    
    # Quick analysis
    analysis = analyze_consolidated_results(consolidated)
    print(f"Final opinion: {analysis['mean_final_opinion']:.3f} ± {analysis['std_final_opinion']:.3f}")
    
    return consolidated, analysis


# Legacy helper functions for backward compatibility
def get_trajectory_replicas(consolidated):
    """Extract trajectory data from consolidated results"""
    if consolidated['opinions'] is not None:
        return {
            'opinions': consolidated['opinions'],
            'save_steps': consolidated['save_steps'],
            'replica_indices': consolidated['trajectory_replica_indices']
        }
    return None


def get_aggregate_data(consolidated):
    """Extract aggregate time series from consolidated results"""
    return {
        'mean_opinion': consolidated['mean_opinion'],
        'opinion_var': consolidated['opinion_var'],
        'S_count': consolidated['S_count'],
        'I_count': consolidated['I_count'],
        'R_count': consolidated['R_count'],
        'fraction_ever_infected': consolidated['fraction_ever_infected']
    }