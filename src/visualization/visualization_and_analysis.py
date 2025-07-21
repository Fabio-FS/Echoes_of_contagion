import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import glob
from matplotlib.colors import TwoSlopeNorm
# Load data


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
def load_and_merge_consolidated(folders):
    """Load consolidated results from multiple folders and merge them"""
    all_consolidated = []
    
    # Load all consolidated files
    for folder in folders:
        pkl_files = glob.glob(os.path.join(folder, "*.pkl"))
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                consolidated = pickle.load(f)
                all_consolidated.append(consolidated)
    
    if not all_consolidated:
        raise ValueError("No consolidated files found")
    
    # Use first result as template
    first = all_consolidated[0]
    n_steps = first['S_count'].shape[1]
    
    # Calculate total replicas
    total_replicas = sum(cons['n_replicas'] for cons in all_consolidated)
    
    # Create merged structure
    merged = {
        'parameters': first['parameters'],  # Assume same parameters
        'n_replicas': total_replicas,
        'trajectory_replica_indices': np.array([]),  # Will update
        
        # Initialize merged arrays
        'S_count': np.zeros((total_replicas, n_steps), dtype=np.int16),
        'I_count': np.zeros((total_replicas, n_steps), dtype=np.int16),
        'R_count': np.zeros((total_replicas, n_steps), dtype=np.int16),
        'mean_opinion': np.zeros((total_replicas, n_steps), dtype=np.float32),
        'opinion_var': np.zeros((total_replicas, n_steps), dtype=np.float32),
        'fraction_ever_infected': np.zeros((total_replicas, n_steps), dtype=np.float32),
        'opinion_bins': np.zeros((total_replicas, n_steps, 20), dtype=np.float32),
        
        'save_steps': first['save_steps'],
        'bin_edges': first['bin_edges'],
        'bin_labels': first['bin_labels']
    }
    
    # Merge trajectory data if available
    total_trajectory_replicas = sum(
        len(cons['trajectory_replica_indices']) for cons in all_consolidated
        if cons['opinions'] is not None
    )
    
    if total_trajectory_replicas > 0:
        n_saves, n_humans = first['opinions'].shape[1:] if first['opinions'] is not None else (0, 0)
        merged['opinions'] = np.zeros((total_trajectory_replicas, n_saves, n_humans), dtype=np.float32)
    else:
        merged['opinions'] = None
    
    # Fill merged arrays
    replica_offset = 0
    traj_offset = 0
    new_traj_indices = []
    
    for consolidated in all_consolidated:
        n_reps = consolidated['n_replicas']
        
        # Copy aggregate data
        merged['S_count'][replica_offset:replica_offset + n_reps] = consolidated['S_count']
        merged['I_count'][replica_offset:replica_offset + n_reps] = consolidated['I_count']
        merged['R_count'][replica_offset:replica_offset + n_reps] = consolidated['R_count']
        merged['mean_opinion'][replica_offset:replica_offset + n_reps] = consolidated['mean_opinion']
        merged['opinion_var'][replica_offset:replica_offset + n_reps] = consolidated['opinion_var']
        merged['fraction_ever_infected'][replica_offset:replica_offset + n_reps] = consolidated['fraction_ever_infected']
        merged['opinion_bins'][replica_offset:replica_offset + n_reps] = consolidated['opinion_bins']
        
        # Copy trajectory data if available
        if consolidated['opinions'] is not None:
            n_traj = consolidated['opinions'].shape[0]
            merged['opinions'][traj_offset:traj_offset + n_traj] = consolidated['opinions']
            
            # Update trajectory indices
            old_indices = consolidated['trajectory_replica_indices']
            new_indices = old_indices + replica_offset
            new_traj_indices.extend(new_indices)
            
            traj_offset += n_traj
        
        replica_offset += n_reps
    
    merged['trajectory_replica_indices'] = np.array(new_traj_indices)
    
    return merged


# Preprocessing data
def get_simple_frequencies(consolidated_results):
    """One-liner to get the frequencies"""
    results = classify_all_replicas_simple(consolidated_results)
    print_simple_summary(results)
    return results['state_fractions']
def print_simple_summary(classification_results):
    """Print restrictive 4-way summary"""
    
    state_names = {
        1: "Extremely negative",
        2: "Extremely positive", 
        3: "Extremely extremist",
        4: "Something else"
    }
    
    print("Restrictive Classification:")
    print("=" * 35)
    
    for state_type in [1, 2, 3, 4]:
        count = classification_results['state_counts'][state_type]
        fraction = classification_results['state_fractions'][state_type]
        print(f"{state_names[state_type]:20s}: {count:2d} ({fraction:5.1%})")
    
    print(f"\nTotal: {classification_results['total_replicas']} replicas")
    
    # Show thresholds for clarity
    print("\nThresholds:")
    print("  Extremely negative: >80% below -0.8")
    print("  Extremely positive: >80% above +0.8") 
    print("  Extremely extremist: >10% above +0.8, >10% below -0.8, <20% in middle")
def classify_all_replicas_simple(consolidated_results):
    """Classify all replicas with simple 4-way system"""
    
    n_replicas = consolidated_results['n_replicas']
    final_bins = consolidated_results['opinion_bins'][:, -1, :]   # Final distributions
    
    results = {}
    state_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for replica in range(n_replicas):
        state_type, description = classify_final_state_simple(final_bins[replica])
        
        results[replica] = {
            'state_type': state_type,
            'description': description
        }
        state_counts[state_type] += 1
    
    # Summary
    state_fractions = {k: v/n_replicas for k, v in state_counts.items()}
    
    return {
        'replica_classifications': results,
        'state_counts': state_counts,
        'state_fractions': state_fractions,
        'total_replicas': n_replicas
    }
def classify_final_state_simple(final_bins, extreme_threshold=0.8):
    """
    Restrictive 4-way classification:
    1) Extremely negative: >80% below -0.8
    2) Extremely positive: >80% above +0.8
    3) Extremely extremist: >10% above +0.8, >10% below -0.8, <20% in middle
    4) Something else
    
    Args:
        final_bins: final opinion distribution (20 bins from -1 to 1)
        extreme_threshold: opinions beyond ±this are "extreme" (default 0.8)
    
    Returns:
        state_type (int), description (str)
    """
    
    # Map bins to regions
    bin_edges = np.arange(-1.0, 1.1, 0.1)  # 20 bins
    extreme_neg_bins = bin_edges[:-1] < -extreme_threshold  # bins < -0.8
    extreme_pos_bins = bin_edges[:-1] > extreme_threshold   # bins > 0.8
    middle_bins = np.abs(bin_edges[:-1]) <= extreme_threshold  # bins within ±0.8
    
    # Calculate fractions in each region
    frac_extreme_neg = np.sum(final_bins[extreme_neg_bins])
    frac_extreme_pos = np.sum(final_bins[extreme_pos_bins])
    frac_middle = np.sum(final_bins[middle_bins])
    
    # Restrictive classification
    if frac_extreme_neg > 0.8:
        return 1, "Extremely negative"
    elif frac_extreme_pos > 0.8:
        return 2, "Extremely positive"
    elif (frac_extreme_neg > 0.1 and frac_extreme_pos > 0.1 and frac_middle < 0.5):
        return 3, "Extremely extremist"
    else:
        return 4, "Something else"
def attack_rate_metric(consolidated_data):
    """Calculate final attack rate (fraction ever infected)"""
    final_infected = consolidated_data['fraction_ever_infected'][:, -1]
    return np.median(final_infected) * 100  # Convert to percentage
def calculate_relative_performance(treatment_data, baseline_data, metric_func, 
                                 param1_name, param2_name):
    """
    Calculate percentage difference between treatment and baseline data.
    
    Args:
        treatment_data: dict with keys as (param1, param2) tuples
        baseline_data: dict with keys as (param1, param2) tuples (baseline reference)
        metric_func: function that takes consolidated data and returns a scalar metric
        param1_name: name of first parameter (e.g., 'threshold')
        param2_name: name of second parameter (e.g., 'n_bots')
    
    Returns:
        diff_array: 2D array of percentage differences
        param1_values: sorted unique values of param1
        param2_values: sorted unique values of param2
        baseline_value: baseline metric value for reference
    """
    
    # Extract unique parameter values
    param1_values = sorted(set(key[0] for key in treatment_data.keys()))
    param2_values = sorted(set(key[1] for key in treatment_data.keys()))
    
    # Calculate baseline metric (use first available baseline data point)
    baseline_key = next(iter(baseline_data.keys()))
    baseline_value = metric_func(baseline_data[baseline_key])
    
    # Initialize result array with NaN
    diff_array = np.full((len(param2_values), len(param1_values)), np.nan)
    
    # Calculate percentage differences
    for i, param2_val in enumerate(param2_values):
        for j, param1_val in enumerate(param1_values):
            key = (param1_val, param2_val)
            if key in treatment_data:
                treatment_value = metric_func(treatment_data[key])
                percent_diff = ((treatment_value - baseline_value) / baseline_value) * 100
                diff_array[i, j] = percent_diff
    
    return diff_array, param1_values, param2_values, baseline_value





# Plotting

def plot_opinion_trajectory(data, who = 0, fig = None, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    time = data["save_steps"]
    subdata = data["opinions"][who]
    L = min(50, data["parameters"]["n_humans"])
    for v in range(L):
        color = value_to_color(subdata[0,v])
        ax.plot(time, subdata[:,v], color=color)
    
    return fig, ax

def plot_attack_rate_comparison(treatment_data, baseline_data, 
                               param1_name='threshold', param2_name='n_bots'):
    """
    Convenience function for attack rate comparison heatmaps.
    """
    # Calculate differences
    diffs, p1_vals, p2_vals, baseline = calculate_relative_performance(
        treatment_data, baseline_data, attack_rate_metric, param1_name, param2_name
    )
    
    # Plot
    fig, ax = plot_heatmap(
        diffs, p1_vals, p2_vals, baseline, param1_name, param2_name,
        baseline_label='attack rate'
    )
    
    return fig, ax, diffs, baseline



def plot_epidemic_heatmap_relative(data_dict, param1_values=None, param2_values=None, 
                                   param1_name="param1", param2_name="param2",
                                   title="Relative Change in Attack Rate from Baseline", figsize=None,
                                   param1_is_rows=True, color_map='RdBu_r', xlabel = None, ylabel = None, fontsize = 10):
    """
    Plot heatmap of attack rates relative to baseline (first row).
    Shows percentage change from baseline, with baseline variability as "noise floor".
    Values within baseline variability range are shown as white (not significant).
    
    Parameters:
    - data_dict: dictionary with parameter tuple keys
    - param1_values: list of values for first parameter 
    - param2_values: list of values for second parameter
    - param1_name: name of first parameter for labels
    - param2_name: name of second parameter for labels
    - title: plot title
    - figsize: figure size (auto-calculated if None)
    - param1_is_rows: if True, param1 varies along rows (y-axis), param2 along columns (x-axis)
    - color_map: divergent colormap (default: 'RdBu_r')
    """
    
    # Auto-detect parameters if not provided
    if param1_values is None or param2_values is None:
        all_keys = list(data_dict.keys())
        if not all_keys:
            print("No data found!")
            return None, None, None, None
            
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
    
    # Initialize matrices
    attack_rate_matrix = np.full((n_rows, n_cols), np.nan)
    baseline_variability = np.full(n_cols, np.nan)
    
    # Fill the attack rate matrix
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
                
                # Calculate variability for baseline row (first row)
                if i == 0:
                    fractions = final_infected_recovered / consolidated['parameters']['n_humans']
                    std_final = np.std(fractions)
                    baseline_variability[j] = std_final
    
    # Calculate baseline values (first row)
    baseline_values = attack_rate_matrix[0, :]
    
    # Calculate relative changes as percentage differences from baseline
    relative_matrix = np.full((n_rows, n_cols), np.nan)
    
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(attack_rate_matrix[i, j]) and not np.isnan(baseline_values[j]):
                if baseline_values[j] > 0:
                    # Percentage change: (new - baseline) / baseline * 100
                    relative_change = ((attack_rate_matrix[i, j] - baseline_values[j]) / baseline_values[j]) * 100
                    relative_matrix[i, j] = relative_change
                else:
                    # Handle case where baseline is 0
                    relative_matrix[i, j] = 0 if attack_rate_matrix[i, j] == 0 else np.inf
    
    # For the baseline row, show deviation from mean baseline
    baseline_mean = np.nanmedian(baseline_values)
    for j in range(n_cols):
        if not np.isnan(baseline_values[j]):
            # Show percentage deviation from median baseline
            deviation_percent = ((baseline_values[j] - baseline_mean) / baseline_mean) * 100
            relative_matrix[0, j] = deviation_percent
    
    # Calculate noise threshold from baseline row deviations
    # Use the 75th percentile of absolute deviations from mean in the baseline row
    baseline_deviations = relative_matrix[0, :][~np.isnan(relative_matrix[0, :])]
    noise_threshold = np.percentile(np.abs(baseline_deviations), 75) if len(baseline_deviations) > 0 else 5.0
    print(f"Noise threshold (75th percentile baseline deviation): ±{noise_threshold:.1f}%")
    
    # Create noise-aware matrix: set values within noise threshold to exactly 0
    noise_aware_matrix = relative_matrix.copy()
    
    # Apply noise threshold to ALL rows (including baseline)
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(noise_aware_matrix[i, j]):
                if abs(noise_aware_matrix[i, j]) <= noise_threshold:
                    noise_aware_matrix[i, j] = 0  # Set to exactly 0 (will be white)
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (max(8, n_cols * 1.2), max(6, n_rows * 1.2))
    
    # Determine color scale limits (more aggressive scaling)
    non_baseline_values = noise_aware_matrix[1:, :].flatten()
    non_baseline_values = non_baseline_values[~np.isnan(non_baseline_values)]
    non_zero_values = non_baseline_values[non_baseline_values != 0]  # Exclude noise-level values
    
    if len(non_zero_values) > 0:
        max_abs_change = np.max(np.abs(non_zero_values))
        # More aggressive scaling: use actual data range rather than fixed ±30
        vmin, vmax = -max_abs_change, max_abs_change
    else:
        max_abs_change = 30  # Default value
        vmin, vmax = -30, 30  # Default range
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(noise_aware_matrix, cmap=color_map, aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([str(val) for val in col_values], fontsize=fontsize)
    ax.set_yticklabels([str(val) for val in row_values], fontsize=fontsize)
    
    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(relative_matrix[i, j]):  # Use original values for text
                if i == 0:
                    # Baseline row: show deviation from median baseline
                    value = relative_matrix[i, j]
                    sign = '+' if value > 0 else ''
                    text = f'{sign}{value:.1f}%'
                    # Check if within noise threshold
                    if abs(value) <= noise_threshold:
                        text_color = 'gray'  # Gray for within noise
                    else:
                        text_color = 'black'  # Black for significant baseline deviations
                else:
                    # Other rows: show relative change
                    value = relative_matrix[i, j]
                    if np.isinf(value):
                        text = '∞'
                        text_color = 'black'
                    else:
                        # Check if this value was set to 0 (within noise)
                        if noise_aware_matrix[i, j] == 0 and value != 0:
                            # Within noise threshold - show in gray
                            sign = '+' if value > 0 else ''
                            text = f'{sign}{value:.1f}%'
                            text_color = 'gray'
                        else:
                            # Significant change - normal formatting
                            sign = '+' if value > 0 else ''
                            text = f'{sign}{value:.1f}%'
                            # Choose text color based on background intensity
                            abs_normalized = abs(noise_aware_matrix[i, j]) / max_abs_change if max_abs_change > 0 else 0
                            text_color = 'white' if abs_normalized > 0.5 else 'black'
                
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontweight='bold', fontsize=10)
            else:
                ax.text(j, i, 'N/A', ha='center', va='center',
                       color='red', fontweight='bold')
    
    # Labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize*1.8)
    else:
        ax.set_xlabel(col_name, fontsize=fontsize*1.8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize*1.8)
    else:
        ax.set_ylabel(row_name, fontsize=fontsize*1.8)
    
    ax.set_title(title, fontsize=fontsize*1.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Change from Baseline (%)', rotation=270, labelpad=20, fontsize=fontsize*1.2)
    
    # Add horizontal line to separate baseline row
    ax.axhline(y=0.5, color='black', linewidth=2, alpha=0.8)
    
    # Add text annotation explaining noise threshold
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, noise_aware_matrix, baseline_values, noise_threshold





# Helper functions

def value_to_color(value, cmap='RdBu'):
    """
    Convert a value between -1 and 1 to a color code.
    
    Parameters:
    - value: float between -1 and 1
    - cmap: colormap name (string) or matplotlib colormap object
    
    Returns:
    - color: RGBA tuple or hex string depending on colormap
    """
    
    # Get colormap if string
    cmap = plt.cm.get_cmap(cmap)
    
    # Normalize value from [-1, 1] to [0, 1]
    normalized = (value + 1) / 2
    
    # Get color
    return cmap(normalized)
def layout(fig, ax, xlim = (0, 10000)):
    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_xlabel("time", fontsize=18)
    ax.set_ylabel("opinion", fontsize=18)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_xticks([0, xlim[1]/2, xlim[1]])
    ax.set_xticklabels([], fontsize=18)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, 0, 1], fontsize=18)















































def plot_heatmap(diff_array, param1_values, param2_values, baseline_value,
                param1_name, param2_name, title=None, figsize=(12, 8), 
                cmap='BrBG_r', vmin=None, vmax=None, annotate=True, 
                fmt='+.1f', baseline_label='attack rate'):
    """
    Plot heatmap of percentage differences from baseline.
    
    Args:
        diff_array: 2D array of percentage differences
        param1_values: values for x-axis (param1)
        param2_values: values for y-axis (param2)
        baseline_value: baseline metric value
        param1_name: label for x-axis
        param2_name: label for y-axis
        title: plot title (auto-generated if None)
        figsize: figure size tuple
        cmap: colormap name
        vmin, vmax: color scale limits (auto if None)
        annotate: whether to show percentage values on cells
        fmt: format string for annotations
        baseline_label: description of the metric
    
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Auto-scale color limits if not provided
    if vmin is None or vmax is None:
        abs_max = np.nanmax(np.abs(diff_array))
        vmin = vmin or -abs_max
        vmax = vmax or abs_max
    
    # Create colormap with white at zero
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Plot heatmap
    im = ax.imshow(diff_array, cmap=cmap, norm=norm, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(param1_values)))
    ax.set_yticks(range(len(param2_values)))
    ax.set_xticklabels(param1_values)
    ax.set_yticklabels(param2_values)
    
    # Labels
    ax.set_xlabel(f"{param1_name.replace('_', ' ').title()}")
    ax.set_ylabel(f"{param2_name.replace('_', ' ').title()}")
    
    # Title
    if title is None:
        title = f"Baseline (random ranking) {baseline_label}: {baseline_value:.0f}%"
    ax.set_title(title)
    
    # Annotations
    if annotate:
        for i in range(len(param2_values)):
            for j in range(len(param1_values)):
                if not np.isnan(diff_array[i, j]):
                    text = f"{diff_array[i, j]:{fmt}}%"
                    ax.text(j, i, text, ha="center", va="center", 
                           color="white" if abs(diff_array[i, j]) > abs_max*0.5 else "black",
                           fontweight="bold" if abs(diff_array[i, j]) > 10 else "normal")
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage change from baseline", rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig, ax


# Example metric functions




























def convergence_time_metric(consolidated_data, extreme_threshold=0.8):
    """
    Calculate median convergence time to extreme states across replicas - VECTORIZED VERSION.
    
    Args:
        consolidated_data: consolidated simulation results
        extreme_threshold: threshold for extreme opinions (default 0.8)
    
    Returns:
        median_convergence_time: median timesteps to reach extreme state (NaN if none converge)
    """
    opinion_bins = consolidated_data['opinion_bins']  # Shape: (n_replicas, n_timesteps, 20)
    n_replicas, n_timesteps, n_bins = opinion_bins.shape
    
    # Vectorized extreme state detection for ALL replicas and timesteps at once
    
    # Map bins to regions (same logic as classify_final_state_simple)
    bin_edges = np.arange(-1.0, 1.1, 0.1)  # 20 bins
    extreme_neg_bins = bin_edges[:-1] < -extreme_threshold  # bins < -0.8
    extreme_pos_bins = bin_edges[:-1] > extreme_threshold   # bins > 0.8
    middle_bins = np.abs(bin_edges[:-1]) <= extreme_threshold  # bins within ±0.8
    
    # Calculate fractions for all replicas and timesteps at once
    # Shape: (n_replicas, n_timesteps)
    frac_extreme_neg = np.sum(opinion_bins[:, :, extreme_neg_bins], axis=2)
    frac_extreme_pos = np.sum(opinion_bins[:, :, extreme_pos_bins], axis=2)
    frac_middle = np.sum(opinion_bins[:, :, middle_bins], axis=2)
    
    # Vectorized extreme state detection
    # Shape: (n_replicas, n_timesteps) - True where extreme state is reached
    is_extreme_neg = frac_extreme_neg > 0.8
    is_extreme_pos = frac_extreme_pos > 0.8
    is_extreme_pol = (frac_extreme_neg > 0.1) & (frac_extreme_pos > 0.1) & (frac_middle < 0.5)
    
    # Combined: any extreme state
    is_extreme = is_extreme_neg | is_extreme_pos | is_extreme_pol
    
    # Find first convergence time for each replica
    convergence_times = np.full(n_replicas, np.nan)
    
    for replica in range(n_replicas):
        extreme_timesteps = np.where(is_extreme[replica, :])[0]
        if len(extreme_timesteps) > 0:
            convergence_times[replica] = extreme_timesteps[0]  # First occurrence
    
    return np.nanmedian(convergence_times)


def calculate_convergence_matrix(data_dict, param1_name, param2_name):
    """
    Calculate convergence time matrix - reuses logic from calculate_relative_performance.
    WITH PROGRESS PRINTING!
    
    Args:
        data_dict: dict with keys as (param1, param2) tuples
        param1_name: name of first parameter (e.g., 'threshold')
        param2_name: name of second parameter (e.g., 'n_bots')
    
    Returns:
        convergence_matrix: 2D array of median convergence times
        param1_values: sorted unique values of param1
        param2_values: sorted unique values of param2
    """
    
    # Extract unique parameter values (reuse from calculate_relative_performance)
    param1_values = sorted(set(key[0] for key in data_dict.keys()))
    param2_values = sorted(set(key[1] for key in data_dict.keys()))
    
    # Initialize result array with NaN
    convergence_matrix = np.full((len(param2_values), len(param1_values)), np.nan)
    
    # Calculate total combinations for progress
    total_combinations = len(param2_values) * len(param1_values)
    current_combination = 0
    
    print(f"Processing {total_combinations} parameter combinations...")
    
    # Calculate convergence times
    for i, param2_val in enumerate(param2_values):
        for j, param1_val in enumerate(param1_values):
            current_combination += 1
            key = (param1_val, param2_val)
            
            if key in data_dict:
                print(f"  {current_combination}/{total_combinations}: {param1_name}={param1_val}, {param2_name}={param2_val}")
                convergence_time = convergence_time_metric(data_dict[key])
                convergence_matrix[i, j] = convergence_time
                print(f"    -> Convergence time: {convergence_time if not np.isnan(convergence_time) else 'Never'}")
            else:
                print(f"  {current_combination}/{total_combinations}: {param1_name}={param1_val}, {param2_name}={param2_val} - MISSING DATA")
    
    print("Done!")
    return convergence_matrix, param1_values, param2_values


def plot_convergence_time_heatmap(data_dict, param1_name='threshold', param2_name='n_bots',
                                 title="Median Convergence Time to Extreme States", 
                                 figsize=None, fontsize=10, xlabel=None, ylabel=None):
    """
    Plot heatmap of convergence times - reuses plotting logic from plot_epidemic_heatmap_relative.
    
    Args:
        data_dict: dictionary with parameter tuple keys
        param1_name: name of first parameter for labels
        param2_name: name of second parameter for labels
        title: plot title
        figsize: figure size (auto-calculated if None)
        fontsize: font size for labels
        xlabel, ylabel: custom axis labels
    
    Returns:
        fig, ax: matplotlib figure and axes objects
        convergence_matrix: 2D array of convergence times
    """
    
    # Calculate convergence matrix
    convergence_matrix, param1_values, param2_values = calculate_convergence_matrix(
        data_dict, param1_name, param2_name
    )
    
    n_rows = len(param2_values)
    n_cols = len(param1_values)
    
    # Auto-calculate figure size (reuse from plot_epidemic_heatmap_relative)
    if figsize is None:
        figsize = (max(8, n_cols * 1.2), max(6, n_rows * 1.2))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color scale limits
    valid_values = convergence_matrix[~np.isnan(convergence_matrix)]
    if len(valid_values) > 0:
        vmin, vmax = np.min(valid_values), np.max(valid_values)
        # Use a colormap where low values (fast convergence) are dark
        cmap = 'viridis_r'  # Reversed viridis: yellow=fast, purple=slow
    else:
        vmin, vmax = 0, 1000
        cmap = 'viridis_r'
    
    # Create heatmap
    im = ax.imshow(convergence_matrix, cmap=cmap, aspect='auto', 
                   vmin=vmin, vmax=vmax, interpolation='nearest')
    
    # Set ticks and labels (reuse from plot_epidemic_heatmap_relative)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([str(val) for val in param1_values], fontsize=fontsize)
    ax.set_yticklabels([str(val) for val in param2_values], fontsize=fontsize)
    
    # Add text annotations
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(convergence_matrix[i, j]):
                value = convergence_matrix[i, j]
                # Format as integer timesteps
                text = f'{int(value)}'
                
                # Choose text color based on background intensity
                normalized_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0
                text_color = 'white' if normalized_value < 0.5 else 'black'  # Light bg = black text
                
                ax.text(j, i, text, ha='center', va='center',
                       color=text_color, fontweight='bold', fontsize=10)
            else:
                ax.text(j, i, 'NC', ha='center', va='center',  # NC = Never Converges
                       color='red', fontweight='bold', fontsize=10)
    
    # Labels and title
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize*1.8)
    else:
        ax.set_xlabel(param1_name, fontsize=fontsize*1.8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize*1.8)
    else:
        ax.set_ylabel(param2_name, fontsize=fontsize*1.8)
    
    ax.set_title(title, fontsize=fontsize*1.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Convergence Time (timesteps)', rotation=270, labelpad=20, fontsize=fontsize*1.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, convergence_matrix


def plot_convergence_comparison(data_dict, param1_name='threshold', param2_name='n_bots'):
    """
    Convenience function for convergence time analysis - similar to plot_attack_rate_comparison.
    
    Args:
        data_dict: dictionary with parameter combinations
        param1_name: name of first parameter
        param2_name: name of second parameter
    
    Returns:
        fig, ax: plot objects
        convergence_matrix: 2D array of convergence times
    """
    
    fig, ax, convergence_matrix = plot_convergence_time_heatmap(
        data_dict, param1_name, param2_name,
        title=f"Median Convergence Time to Extreme States"
    )
    
    return fig, ax, convergence_matrix





def print_average_across_run_binned(data):
    N_steps, N_bins = data["opinion_bins"][0].shape
    RES = np.zeros((N_steps,N_bins))

    for i in range(N_steps):
        for j in range(N_bins):
            RES[i,j] = np.mean(data["opinion_bins"][:,i,j])
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(RES.T,vmin = 0, vmax = 0.4, 
                   cmap = "viridis_r", aspect = "auto", 
                   interpolation = "none",
                   origin = "lower")
    fig.colorbar(im)
    ax.set_xlabel("Time", fontsize = 18)
    ax.set_ylabel("Opinion", fontsize = 18)

    ax.set_xticks([0,N_steps/2,N_steps])
    ax.set_xticklabels([0,N_steps/2,N_steps], fontsize = 18)
    ax.set_yticks([0,10,19])
    ax.set_yticklabels([-1,0,1], fontsize = 18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0,N_steps)
    plt.show()
    return RES