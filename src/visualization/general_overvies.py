import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_consolidated_overview(consolidated, figsize=(15, 10)):
    """
    Create 2x3 overview plot:
    [0,0]: Heatmap of opinion bins averaged across all 100 runs
    [0,1] to [1,2]: Individual agent trajectories for first 5 replicas
    
    Args:
        consolidated: Consolidated results from simulations()
        figsize: Figure size tuple
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # [0,0]: Heatmap of binned opinions averaged across all runs
    ax_heatmap = axes[0, 0]
    
    # Average opinion bins across all replicas (shape: n_steps, n_bins)
    avg_opinion_bins = np.mean(consolidated['opinion_bins'], axis=0)
    
    # Create heatmap
    im = ax_heatmap.imshow(avg_opinion_bins.T, aspect='auto', origin='lower',
                          extent=[0, avg_opinion_bins.shape[0]-1, 0, avg_opinion_bins.shape[1]-1],
                          cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label('Fraction of Agents', rotation=270, labelpad=15)
    
    ax_heatmap.set_xlabel('Time Step')
    ax_heatmap.set_ylabel('Opinion Bin')
    ax_heatmap.set_title('Average Opinion Distribution\n(All 100 Replicas)')
    
    # Set y-axis labels to show bin ranges
    bin_edges = consolidated['bin_edges']
    n_bins = len(bin_edges) - 1
    tick_positions = np.arange(0, n_bins, 2)  # Show every other bin to avoid crowding
    tick_labels = [f'[{bin_edges[i]:.1f},{bin_edges[i+1]:.1f})' for i in tick_positions]
    ax_heatmap.set_yticks(tick_positions)
    ax_heatmap.set_yticklabels(tick_labels, fontsize=8)
    
    # [0,1] to [1,2]: Individual trajectory plots for first 5 replicas
    subplot_positions = [(0,1), (0,2), (1,0), (1,1), (1,2)]
    
    # Check if we have trajectory data
    if consolidated['opinions'] is None:
        for i, pos in enumerate(subplot_positions):
            axes[pos].text(0.5, 0.5, 'No trajectory\ndata available', 
                          ha='center', va='center', transform=axes[pos].transAxes,
                          fontsize=12, color='red')
            axes[pos].set_title(f'Replica {i+1}')
        plt.tight_layout()
        return fig, axes
    
    # Plot trajectories for available replicas
    save_steps = consolidated['save_steps']
    n_trajectory_replicas = consolidated['opinions'].shape[0]
    
    for i, pos in enumerate(subplot_positions):
        ax = axes[pos]
        
        if i < n_trajectory_replicas:
            # Plot all agent trajectories for this replica
            opinions = consolidated['opinions'][i]  # Shape: (n_time_points, n_humans)
            
            for agent_id in range(opinions.shape[1]):
                ax.plot(save_steps, opinions[:, agent_id], alpha=0.3, linewidth=0.5)
            
            ax.set_ylim(-1, 1)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Opinion')
            ax.set_title(f'Replica {i+1}: All Agent Trajectories')
            ax.grid(True, alpha=0.3)
            
        else:
            # No data for this replica
            ax.text(0.5, 0.5, f'Replica {i+1}\nNot available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            ax.set_title(f'Replica {i+1}')
    
    plt.tight_layout()
    return fig, axes


def plot_bin_evolution_detailed(consolidated, figsize=(12, 8)):
    """
    Detailed view of how specific opinion bins evolve over time
    Shows extreme bins (-1 to -0.8, -0.1 to 0.1, 0.8 to 1) separately
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Average across all replicas
    avg_bins = np.mean(consolidated['opinion_bins'], axis=0)  # Shape: (n_steps, n_bins)
    time_steps = np.arange(avg_bins.shape[0])
    
    # Define bins of interest (assuming 20 bins from -1 to 1)
    bin_edges = consolidated['bin_edges']
    
    # Find indices for extreme negative, center, and extreme positive
    extreme_neg_idx = 0  # First bin: -1.0 to -0.9
    center_neg_idx = 9   # Around -0.1 to 0.0
    center_pos_idx = 10  # Around 0.0 to 0.1  
    extreme_pos_idx = -1 # Last bin: 0.9 to 1.0
    
    # Plot evolution of key bins
    ax.plot(time_steps, avg_bins[:, extreme_neg_idx], 'r-', linewidth=2, 
           label=f'Extreme Negative [{bin_edges[0]:.1f}, {bin_edges[1]:.1f})')
    ax.plot(time_steps, avg_bins[:, center_neg_idx], 'orange', linewidth=2,
           label=f'Center Negative [{bin_edges[9]:.1f}, {bin_edges[10]:.1f})')
    ax.plot(time_steps, avg_bins[:, center_pos_idx], 'lightblue', linewidth=2,
           label=f'Center Positive [{bin_edges[10]:.1f}, {bin_edges[11]:.1f})')
    ax.plot(time_steps, avg_bins[:, extreme_pos_idx], 'b-', linewidth=2,
           label=f'Extreme Positive [{bin_edges[-2]:.1f}, {bin_edges[-1]:.1f})')
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Fraction of Agents')
    ax.set_title('Evolution of Key Opinion Bins (Average across all replicas)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


def plot_heatmap_grid(data_dict, param1_values=None, param2_values=None, 
                     param1_name="param1", param2_name="param2",
                     title="Opinion Distribution Evolution", figsize=None,
                     param1_is_rows=True, max_time_steps=None):
    """
    Generic function to plot heatmaps of opinion bin trajectories in a grid.
    
    Parameters:
    - data_dict: dictionary with parameter tuple keys
    - param1_values: list of values for first parameter 
    - param2_values: list of values for second parameter
    - param1_name: name of first parameter for labels
    - param2_name: name of second parameter for labels
    - title: plot title
    - figsize: figure size (auto-calculated if None)
    - param1_is_rows: if True, param1 varies along rows (y-axis), param2 along columns (x-axis)
                     if False, param1 varies along columns (x-axis), param2 along rows (y-axis)
    - max_time_steps: if None, show all time steps; if int, show only first N time steps
    
    Examples:
    # Show only first 1000 time steps
    plot_heatmap_grid(ic_data, max_time_steps=1000,
                     param1_values=[0.1, 0.5],      
                     param2_values=[-0.9, 0, 0.9],  
                     param1_name="var", param2_name="mean")
    """
    
    # Auto-detect parameters if not provided
    if param1_values is None or param2_values is None:
        # Extract all unique parameter values from keys
        all_keys = list(data_dict.keys())
        if not all_keys:
            print("No data found!")
            return None, None
            
        # Handle single parameter case (keys are not tuples)
        if not isinstance(all_keys[0], tuple):
            if param1_values is None:
                param1_values = sorted(set(all_keys))
            if param2_values is None:
                param2_values = [None]  # Single column
        else:
            # Handle multi-parameter case
            if param1_values is None:
                param1_values = sorted(set(key[0] for key in all_keys))
            if param2_values is None:
                if len(all_keys[0]) >= 2:
                    param2_values = sorted(set(key[1] for key in all_keys))
                else:
                    param2_values = [None]  # Single column
    
    # Determine which values go on rows vs columns
    if param1_is_rows:
        row_values = param1_values
        col_values = param2_values
        row_name = param1_name
        col_name = param2_name
    else:
        row_values = param2_values
        col_values = param1_values
        row_name = param2_name
        col_name = param1_name
    
    # Handle single-element lists
    if len(row_values) == 1:
        n_rows = 1
    else:
        n_rows = len(row_values)
        
    if len(col_values) == 1:
        n_cols = 1
    else:
        n_cols = len(col_values)
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (3 * n_cols + 2, 3 * n_rows + 1)
    
    # Create subplots - handle single subplot case
    if n_rows == 1 and n_cols == 1:
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs = [[axs]]  # Make it 2D for consistent indexing
    elif n_rows == 1:
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        axs = [axs] if n_cols > 1 else [[axs]]  # Make it 2D
    elif n_cols == 1:
        fig, axs = plt.subplots(n_rows, 1, figsize=figsize)
        axs = [[ax] for ax in axs] if n_rows > 1 else [[axs]]  # Make it 2D
    else:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axs = [axs]
        if n_cols == 1:
            axs = [[ax] for ax in axs]
    
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axs[i][j]
            
            # Create key for data lookup
            # Since your data keys are (mean, var), we need to construct them correctly
            if col_val is None:
                key = row_val
            elif row_val is None:
                key = col_val
            else:
                # Determine what row_val and col_val represent based on param1_is_rows
                if param1_is_rows:
                    # param1 (rows) = var, param2 (cols) = mean
                    var_val = row_val
                    mean_val = col_val
                else:
                    # param1 (cols) = mean, param2 (rows) = var  
                    mean_val = row_val
                    var_val = col_val
                
                # Always construct key as (mean, var) since that's how your data is stored
                if param1_is_rows:
                    key = (row_val, col_val)  # (th, nb)
                else:
                    key = (col_val, row_val)  # (nb, th) if flipped
            
            if key is not None and key in data_dict:
                consolidated = data_dict[key]
                
                # Calculate polarization variance
                polarization_var = calculate_polarization_variance(consolidated)
                
                # Calculate average opinion bins across replicas
                avg_opinion_bins = np.mean(consolidated['opinion_bins'], axis=0)
                
                # Apply time step limit if specified
                if max_time_steps is not None:
                    avg_opinion_bins = avg_opinion_bins[:max_time_steps, :]
                    time_extent_max = max_time_steps - 1
                    time_title_suffix = f" (first {max_time_steps} steps)"
                else:
                    time_extent_max = avg_opinion_bins.shape[0] - 1
                    time_title_suffix = ""
                
                # Create heatmap
                im = ax.imshow(avg_opinion_bins.T, aspect='auto', origin='lower',
                              extent=[0, time_extent_max, 0, avg_opinion_bins.shape[1]-1],
                              cmap='viridis', interpolation='none', vmin=0, vmax=0.35)
                
                # Add colorbar for rightmost plots
                if j == n_cols - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Fraction of Agents', rotation=270, labelpad=15)
                
                # Set title
                if col_val is None:
                    title_str = f'{row_name} = {row_val}\npolarization var = {polarization_var:.3f}{time_title_suffix}'
                elif row_val is None:
                    title_str = f'{col_name} = {col_val}\npolarization var = {polarization_var:.3f}{time_title_suffix}'
                else:
                    title_str = f'{row_name} = {row_val}, {col_name} = {col_val}\npolarization var = {polarization_var:.3f}{time_title_suffix}'
                ax.set_title(title_str, fontsize=10)
                
                # Set y-axis labels to show bin ranges
                bin_edges = consolidated['bin_edges']
                n_bins = len(bin_edges) - 1
                tick_positions = np.arange(0, n_bins, 2)  # Show every other bin
                tick_labels = [f'[{bin_edges[i]:.1f},{bin_edges[i+1]:.1f})' for i in tick_positions]
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=8)
            
            else:
                ax.text(0.5, 0.5, f'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Labels
            if i == n_rows - 1:
                ax.set_xlabel('Time Step')
            if j == 0:
                ax.set_ylabel('Opinion Bin')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    return fig, axs



def calculate_polarization_variance(consolidated):
    """
    Calculate the variance of the meta-distribution (averaged across replicas).
    This measures true polarization of the consensus outcome.
    """
    # Get bin centers from bin edges
    bin_edges = consolidated['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get the final time step averaged distribution across all replicas
    final_avg_distribution = np.mean(consolidated['opinion_bins'][:, -1, :], axis=0)
    
    # Calculate variance of this meta-distribution
    # Var(X) = E[X²] - (E[X])²
    mean_opinion = np.sum(final_avg_distribution * bin_centers)
    mean_squared = np.sum(final_avg_distribution * (bin_centers**2))
    
    polarization_variance = mean_squared - mean_opinion**2
    
    return polarization_variance

def load_simulation_data(data_dir, file_pattern, param_names, date_suffix="None"):
    """
    Flexible loader for simulation pickle files with variable parameters.
    
    Parameters:
    - data_dir: directory containing the files
    - file_pattern: base pattern (e.g., "results_mean*_var*", "results_ce*_th*_nb*")
    - param_names: list of parameter names in order (e.g., ["mean", "var"] or ["ce", "th", "nb"])
    - date_suffix: date part of filename (default: "2025_07_09"). If None, matches any .pkl file
    
    Returns:
    - Dictionary with parameter tuples as keys and loaded data as values
    
    Examples:
    # IC consensus (2 params) with specific date
    ic_data = load_simulation_data(
        data_dir=r"C:\path\to\IC_consensus",
        file_pattern="results_mean*_var*",
        param_names=["mean", "var"]
    )
    
    # Bot experiment with any date
    bot_data = load_simulation_data(
        data_dir=r"C:\path\to\bot_experiment", 
        file_pattern="results_ce*_th*_nb*",
        param_names=["ce", "th", "nb"],
        date_suffix=None
    )
    
    # Single parameter experiment
    single_data = load_simulation_data(
        data_dir=r"C:\path\to\single_param",
        file_pattern="results_sigma*",
        param_names=["sigma"]
    )
    """
    
    # Validate inputs
    if len(param_names) > 3:
        raise ValueError("Maximum 3 parameters supported")
    if len(param_names) == 0:
        raise ValueError("At least 1 parameter name required")
    
    # Build full pattern
    if date_suffix is None:
        full_pattern = os.path.join(data_dir, f"{file_pattern}.pkl")
    else:
        full_pattern = os.path.join(data_dir, f"{file_pattern}_{date_suffix}.pkl")
    
    files = glob.glob(full_pattern)
    
    if not files:
        print(f"Warning: No files found matching pattern: {full_pattern}")
        return {}
    
    data_dict = {}
    
    for filepath in files:
        try:
            # Extract filename without path and extension
            filename = os.path.basename(filepath)
            
            # Remove .pkl extension
            filename_no_ext = filename.replace(".pkl", "")
            
            # Remove date suffix if specified
            if date_suffix is not None:
                core = filename_no_ext.replace(f"_{date_suffix}", "")
            else:
                # For date_suffix=None, remove any trailing date-like pattern
                # Assumes date is at the end in format YYYY_MM_DD
                parts_temp = filename_no_ext.split('_')
                # Check if last 3 parts look like a date (all numeric)
                if (len(parts_temp) >= 3 and 
                    parts_temp[-3].isdigit() and len(parts_temp[-3]) == 4 and  # year
                    parts_temp[-2].isdigit() and len(parts_temp[-2]) <= 2 and   # month
                    parts_temp[-1].isdigit() and len(parts_temp[-1]) <= 2):     # day
                    core = '_'.join(parts_temp[:-3])
                else:
                    core = filename_no_ext
            
            # Split by underscore and extract parameter parts
            parts = core.split('_')[1:]  # Skip "results" prefix
            
            # Extract parameter values
            param_values = []
            for i, param_name in enumerate(param_names):
                if i >= len(parts):
                    raise ValueError(f"Missing parameter {param_name} in filename {filename}")
                
                # Remove parameter prefix (e.g., "mean0.5" -> "0.5")
                param_str = parts[i][len(param_name):]
                param_val = float(param_str)
                param_values.append(param_val)
            
            # Load the file
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Create key (single value or tuple)
            if len(param_values) == 1:
                key = param_values[0]
            else:
                key = tuple(param_values)
            
            # Store data
            data_dict[key] = data
            
            # Print progress
            if len(param_values) == 1:
                print(f"Loaded: {param_names[0]}={param_values[0]}")
            else:
                param_str = ", ".join([f"{name}={val}" for name, val in zip(param_names, param_values)])
                print(f"Loaded: {param_str}")
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    print(f"\nLoaded {len(data_dict)} files total")
    return data_dict

def plot_epidemic_grid(data_dict, param1_values=None, param2_values=None, 
                      param1_name="param1", param2_name="param2",
                      title="Epidemic Curves: E[I(t) + R(t)]", figsize=None,
                      param1_is_rows=True, max_time_steps=None):
    """
    Generic function to plot epidemic curves E[I(t) + R(t)] in a grid.
    Same signature as plot_heatmap_grid but shows infection curves instead.
    
    Parameters:
    - data_dict: dictionary with parameter tuple keys
    - param1_values: list of values for first parameter 
    - param2_values: list of values for second parameter
    - param1_name: name of first parameter for labels
    - param2_name: name of second parameter for labels
    - title: plot title
    - figsize: figure size (auto-calculated if None)
    - param1_is_rows: if True, param1 varies along rows (y-axis), param2 along columns (x-axis)
    - max_time_steps: if None, show all time steps; if int, show only first N time steps
    """
    
    # Auto-detect parameters if not provided
    if param1_values is None or param2_values is None:
        all_keys = list(data_dict.keys())
        if not all_keys:
            print("No data found!")
            return None, None
            
        if not isinstance(all_keys[0], tuple):
            if param1_values is None:
                param1_values = sorted(set(all_keys))
            if param2_values is None:
                param2_values = [None]
        else:
            if param1_values is None:
                param1_values = sorted(set(key[0] for key in all_keys))
            if param2_values is None:
                if len(all_keys[0]) >= 2:
                    param2_values = sorted(set(key[1] for key in all_keys))
                else:
                    param2_values = [None]
    
    # Determine which values go on rows vs columns
    if param1_is_rows:
        row_values = param1_values
        col_values = param2_values
        row_name = param1_name
        col_name = param2_name
    else:
        row_values = param2_values
        col_values = param1_values
        row_name = param2_name
        col_name = param1_name
    
    # Handle single-element lists
    n_rows = len(row_values) if len(row_values) > 1 else 1
    n_cols = len(col_values) if len(col_values) > 1 else 1
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (3 * n_cols + 2, 3 * n_rows + 1)
    
    # Create subplots - handle single subplot case
    if n_rows == 1 and n_cols == 1:
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        axs = [[axs]]
    elif n_rows == 1:
        fig, axs = plt.subplots(1, n_cols, figsize=figsize)
        axs = [axs] if n_cols > 1 else [[axs]]
    elif n_cols == 1:
        fig, axs = plt.subplots(n_rows, 1, figsize=figsize)
        axs = [[ax] for ax in axs] if n_rows > 1 else [[axs]]
    else:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axs = [axs]
        if n_cols == 1:
            axs = [[ax] for ax in axs]
    
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axs[i][j]
            
            # Create key for data lookup
            if col_val is None:
                key = row_val
            elif row_val is None:
                key = col_val
            else:
                if param1_is_rows:
                    key = (row_val, col_val)
                else:
                    key = (col_val, row_val)
            
            if key is not None and key in data_dict:
                consolidated = data_dict[key]
                
                # Calculate E[I(t) + R(t)] across replicas
                infected_plus_recovered = consolidated['I_count'] + consolidated['R_count']
                
                # Calculate median and quartiles instead of mean ± std
                median_infected_recovered = np.median(infected_plus_recovered, axis=0)
                q1_infected_recovered = np.percentile(infected_plus_recovered, 25, axis=0)
                q3_infected_recovered = np.percentile(infected_plus_recovered, 75, axis=0)
                
                # Apply time step limit if specified
                if max_time_steps is not None:
                    median_infected_recovered = median_infected_recovered[:max_time_steps]
                    q1_infected_recovered = q1_infected_recovered[:max_time_steps]
                    q3_infected_recovered = q3_infected_recovered[:max_time_steps]
                    time_title_suffix = f" (first {max_time_steps} steps)"
                else:
                    time_title_suffix = ""
                
                # Time steps
                t = np.arange(len(median_infected_recovered))
                
                # Plot median curve
                ax.plot(t, median_infected_recovered, 'b-', linewidth=2, label='Median')
                
                # Add interquartile range (Q1 to Q3)
                ax.fill_between(t, q1_infected_recovered, q3_infected_recovered,
                               alpha=0.3, color='blue', label='IQR (Q1-Q3)')
                
                # Final attack rate (percentage) - use the actual final median value shown
                final_attack_rate = median_infected_recovered[-1] / consolidated['parameters']['n_humans'] * 100
                
                # Set title
                if col_val is None:
                    title_str = f'{row_name} = {row_val}\nFinal: {final_attack_rate:.1f}%{time_title_suffix}'
                elif row_val is None:
                    title_str = f'{col_name} = {col_val}\nFinal: {final_attack_rate:.1f}%{time_title_suffix}'
                else:
                    title_str = f'{row_name} = {row_val}, {col_name} = {col_val}\nFinal: {final_attack_rate:.1f}%{time_title_suffix}'
                ax.set_title(title_str, fontsize=10)
                
                ax.set_ylim(0, consolidated['parameters']['n_humans'])
                ax.grid(True, alpha=0.3)
                
            else:
                ax.text(0.5, 0.5, f'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            
            # Labels
            if i == n_rows - 1:
                ax.set_xlabel('Time Step')
            if j == 0:
                ax.set_ylabel('Infected + Recovered')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_epidemic_heatmap(data_dict, param1_values=None, param2_values=None, 
                         param1_name="param1", param2_name="param2",
                         title="Final Attack Rate: Median Fraction Infected", figsize=None,
                         param1_is_rows=True):
    """
    Plot heatmap of final attack rates (median fraction infected at equilibrium).
    
    Parameters:
    - data_dict: dictionary with parameter tuple keys
    - param1_values: list of values for first parameter 
    - param2_values: list of values for second parameter
    - param1_name: name of first parameter for labels
    - param2_name: name of second parameter for labels
    - title: plot title
    - figsize: figure size (auto-calculated if None)
    - param1_is_rows: if True, param1 varies along rows (y-axis), param2 along columns (x-axis)
    """
    
    # Auto-detect parameters if not provided
    if param1_values is None or param2_values is None:
        all_keys = list(data_dict.keys())
        if not all_keys:
            print("No data found!")
            return None, None
            
        if not isinstance(all_keys[0], tuple):
            if param1_values is None:
                param1_values = sorted(set(all_keys))
            if param2_values is None:
                param2_values = [None]
        else:
            if param1_values is None:
                param1_values = sorted(set(key[0] for key in all_keys))
            if param2_values is None:
                if len(all_keys[0]) >= 2:
                    param2_values = sorted(set(key[1] for key in all_keys))
                else:
                    param2_values = [None]
    
    # Determine which values go on rows vs columns
    if param1_is_rows:
        row_values = param1_values
        col_values = param2_values
        row_name = param1_name
        col_name = param2_name
    else:
        row_values = param2_values
        col_values = param1_values
        row_name = param2_name
        col_name = param1_name
    
    n_rows = len(row_values)
    n_cols = len(col_values)
    
    # Initialize matrix to store final attack rates
    attack_rate_matrix = np.full((n_rows, n_cols), np.nan)
    
    # Fill the matrix
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            
            # Create key for data lookup
            if col_val is None:
                key = row_val
            elif row_val is None:
                key = col_val
            else:
                if param1_is_rows:
                    key = (row_val, col_val)
                else:
                    key = (col_val, row_val)
            
            if key is not None and key in data_dict:
                consolidated = data_dict[key]
                
                # Calculate final attack rate (median across replicas)
                infected_plus_recovered = consolidated['I_count'] + consolidated['R_count']
                final_infected_recovered = infected_plus_recovered[:, -1]  # Last time step for each replica
                median_final = np.median(final_infected_recovered)
                
                # Convert to fraction
                attack_rate_fraction = median_final / consolidated['parameters']['n_humans']
                attack_rate_matrix[i, j] = attack_rate_fraction
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (max(8, n_cols * 1.2), max(6, n_rows * 1.2))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attack_rate_matrix, cmap='viridis', aspect='auto', 
                   vmin=0, vmax=1, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([str(val) for val in col_values])
    ax.set_yticklabels([str(val) for val in row_values])
    
    # Add text annotations with percentages
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(attack_rate_matrix[i, j]):
                percentage = attack_rate_matrix[i, j] * 100
                text_color = 'white' if attack_rate_matrix[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{percentage:.1f}%', ha='center', va='center',
                       color=text_color, fontweight='bold')
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                       color='red', fontweight='bold')
    
    # Labels and title
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Final Attack Rate (Fraction)', rotation=270, labelpad=20)
    
    # Format colorbar ticks as percentages
    cbar_ticks = np.arange(0, 1.1, 0.2)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{tick*100:.0f}%' for tick in cbar_ticks])
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, attack_rate_matrix






















