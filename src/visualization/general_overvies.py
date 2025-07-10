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