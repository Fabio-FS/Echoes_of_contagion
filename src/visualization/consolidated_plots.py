import numpy as np
import matplotlib.pyplot as plt

#from general_overvies import calculate_polarization_variance
def calculate_polarization_metrics(consolidated):
    """
    Calculate both meta-distribution variance and average individual variance.
    
    Returns:
    - meta_variance: variance of the averaged final distribution (consensus measure)
    - avg_individual_variance: average variance across individual runs (typical outcome)
    """
    # Get bin centers from bin edges
    bin_edges = consolidated['bin_edges']
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get final distributions for all replicas
    final_distributions = consolidated['opinion_bins'][:, -1, :]  # (n_replicas, n_bins)
    
    # METHOD 1: Meta-distribution variance (your current approach)
    # Average across replicas first, then calculate variance
    final_avg_distribution = np.mean(final_distributions, axis=0)
    mean_opinion_meta = np.sum(final_avg_distribution * bin_centers)
    mean_squared_meta = np.sum(final_avg_distribution * (bin_centers**2))
    meta_variance = mean_squared_meta - mean_opinion_meta**2
    
    # METHOD 2: Average of individual variances
    # Calculate variance for each replica separately, then average
    individual_variances = []
    for replica_dist in final_distributions:
        mean_opinion_indiv = np.sum(replica_dist * bin_centers)
        mean_squared_indiv = np.sum(replica_dist * (bin_centers**2))
        indiv_variance = mean_squared_indiv - mean_opinion_indiv**2
        individual_variances.append(indiv_variance)
    
    avg_individual_variance = np.mean(individual_variances)
    
    return meta_variance, avg_individual_variance


# Updated prepare function to use both metrics
def prepare_heatmap_grid_enhanced(data_dict, param1_values, param2_values, 
                                 param1_name="param1", param2_name="param2",
                                 param1_is_rows=True, max_time_steps=None, 
                                 polarization_metric="meta"):
    """
    Enhanced version that can use either polarization metric.
    
    Parameters:
    - polarization_metric: "meta" (default), "individual", or "both"
    """
    # Determine grid layout
    if param1_is_rows:
        row_values, col_values = param1_values, param2_values
        row_name, col_name = param1_name, param2_name
    else:
        row_values, col_values = param2_values, param1_values
        row_name, col_name = param2_name, param1_name
    
    n_rows, n_cols = len(row_values), len(col_values)
    
    # Initialize grid arrays
    grid_data = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    grid_titles = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Fill grid
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            # Create key for data lookup
            if param1_is_rows:
                key = (row_val, col_val)
            else:
                key = (col_val, row_val)
            
            if key in data_dict:
                data = data_dict[key]
                
                # Handle different data types
                if isinstance(data, dict):
                    # Full consolidated structure
                    meta_var, indiv_var = calculate_polarization_metrics(data)
                    avg_opinion_bins = np.mean(data['opinion_bins'], axis=0)
                    
                    # Choose which metric to display
                    if polarization_metric == "meta":
                        pol_var = meta_var
                        pol_label = "meta_var"
                    elif polarization_metric == "individual":
                        pol_var = indiv_var
                        pol_label = "avg_var"
                    else:  # "both"
                        pol_var = (meta_var, indiv_var)
                        pol_label = "meta/avg"
                        
                elif isinstance(data, np.ndarray):
                    # Raw array - assume it's already averaged opinion_bins
                    if data.ndim == 3:  # (replicas, time, bins)
                        avg_opinion_bins = np.mean(data, axis=0)
                    else:  # (time, bins)
                        avg_opinion_bins = data
                    pol_var = 0.0
                    pol_label = "unknown"
                else:
                    continue
                
                # Apply time step limit if specified
                if max_time_steps is not None:
                    avg_opinion_bins = avg_opinion_bins[:max_time_steps, :]
                
                # Store data (transpose for plotting: bins x time)
                grid_data[i][j] = avg_opinion_bins.T
                
                # Create title
                if isinstance(pol_var, tuple):
                    grid_titles[i][j] = f'{row_name}={row_val}, {col_name}={col_val}\n{pol_label}={pol_var[0]:.3f}/{pol_var[1]:.3f}'
                elif pol_var > 0:
                    grid_titles[i][j] = f'{row_name}={row_val}, {col_name}={col_val}\n{pol_label}={pol_var:.3f}'
                else:
                    grid_titles[i][j] = f'{row_name}={row_val}, {col_name}={col_val}'
    
    return grid_data, grid_titles, (n_rows, n_cols)

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


def prepare_heatmap_grid(data_dict, param1_values, param2_values, 
                        param1_name="param1", param2_name="param2",
                        param1_is_rows=True, max_time_steps=None):
    """
    Prepare data for grid plotting by organizing it into a 2D structure.
    
    Returns:
    - grid_data: 2D list where grid_data[i][j] contains the heatmap array for subplot (i,j)
    - grid_titles: 2D list with title strings for each subplot
    - grid_shape: (n_rows, n_cols)
    """
    # Determine grid layout
    if param1_is_rows:
        row_values, col_values = param1_values, param2_values
        row_name, col_name = param1_name, param2_name
    else:
        row_values, col_values = param2_values, param1_values
        row_name, col_name = param2_name, param1_name
    
    n_rows, n_cols = len(row_values), len(col_values)
    
    # Initialize grid arrays
    grid_data = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    grid_titles = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Fill grid
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            # Create key for data lookup
            if param1_is_rows:
                key = (row_val, col_val)
            else:
                key = (col_val, row_val)
            
            if key in data_dict:
                consolidated = data_dict[key]
                
                # Calculate polarization variance
                polarization_var = calculate_polarization_variance(consolidated)
                
                # Calculate average opinion bins across replicas
                avg_opinion_bins = np.mean(consolidated['opinion_bins'], axis=0)
                
                # Apply time step limit if specified
                if max_time_steps is not None:
                    avg_opinion_bins = avg_opinion_bins[:max_time_steps, :]
                
                # Store data (transpose for plotting: bins x time)
                grid_data[i][j] = avg_opinion_bins.T
                
                # Create title
                grid_titles[i][j] = f'{row_name}={row_val}, {col_name}={col_val}\npol_var={polarization_var:.3f}'
    
    return grid_data, grid_titles, (n_rows, n_cols)


def plot_heatmap_grid(grid_data, grid_titles, grid_shape,
                     title="Opinion Distribution Evolution", 
                     figsize=None, colorbar=False, 
                     custom_xticks=None, custom_yticks=None):
    """
    Plot pre-organized heatmap data in a grid.
    
    Parameters:
    - grid_data: 2D list of arrays (bins x time) or None for missing data
    - grid_titles: 2D list of title strings
    - grid_shape: (n_rows, n_cols)
    - custom_xticks: (positions, labels) for x-axis
    - custom_yticks: (positions, labels) for y-axis  
    """
    n_rows, n_cols = grid_shape
    
    # Auto-calculate figure size
    if figsize is None:
        figsize = (3 * n_cols + 2, 3 * n_rows + 1)
    
    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axs[i, j]
            data = grid_data[i][j]
            
            if data is not None:
                # Create heatmap
                im = ax.imshow(data, aspect='auto', origin='lower',
                              cmap='viridis', interpolation='none', 
                              vmin=0, vmax=0.35)
                
                # Add colorbar if requested
                if colorbar and j == n_cols - 1:
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Fraction of Agents', rotation=270, labelpad=15)
                
                # Set title
                if grid_titles[i][j]:
                    ax.set_title(grid_titles[i][j], fontsize=10)
                
                # Set custom ticks if provided
                if custom_xticks:
                    positions, labels = custom_xticks
                    ax.set_xticks(positions)
                    ax.set_xticklabels(labels)
                    
                if custom_yticks:
                    positions, labels = custom_yticks
                    ax.set_yticks(positions)
                    ax.set_yticklabels(labels)
                
            else:
                # Missing data
                ax.text(0.5, 0.5, 'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            
            # Labels
            if i == n_rows - 1:
                ax.set_xlabel('Time Step')
            if j == 0:
                ax.set_ylabel('Opinion Bin')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    return fig, axs



def plot_single_heatmap(data, title=None, figsize=(8, 6), 
                        custom_xticks=None, custom_yticks=None, 
                        colorbar=True, ax=None):
    """
    Plot a single heatmap.
    
    Parameters:
    - data: 2D array (bins, time) - the heatmap data
    - title: plot title
    - figsize: figure size (ignored if ax provided)
    - custom_xticks: (positions, labels) for x-axis
    - custom_yticks: (positions, labels) for y-axis  
    - colorbar: whether to show colorbar
    - ax: existing axis to plot on (if None, creates new figure)
    """
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create heatmap
    im = ax.imshow(data, aspect='auto', origin='lower',
                  cmap='viridis', interpolation='none', 
                  vmin=0, vmax=0.35)
    
    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Fraction of Agents', rotation=270, labelpad=15)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=12)
    
    # Set custom ticks
    if custom_xticks:
        positions, labels = custom_xticks
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        
    if custom_yticks:
        positions, labels = custom_yticks
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
    
    # Labels
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Opinion Bin')
    
    plt.tight_layout()
    return fig, ax




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