import matplotlib.pyplot as plt
import numpy as np

def plot_observables(RES, param=None, figsize=(12, 8)):
    """
    Plot key observables over time from simulation results.
    
    Parameters:
    - RES: results dictionary from single_simulations()
    - param: parameter dictionary (optional, for title info)
    - figsize: figure size
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Simulation Observables Over Time', fontsize=14)
    
    # Time steps
    t = np.arange(len(RES['mean_opinion']))
    
    # Plot 1: Mean Opinion
    axes[0,0].plot(t, RES['mean_opinion'], 'b-', linewidth=2)
    axes[0,0].set_ylabel('Mean Opinion')
    axes[0,0].set_title('Average Opinion')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 2: Opinion Variance (Polarization)
    axes[0,1].plot(t, RES['opinion_var'], 'r-', linewidth=2)
    axes[0,1].set_ylabel('Opinion Variance')
    axes[0,1].set_title('Opinion Polarization')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Fraction Ever Infected (Cumulative)
    axes[1,0].plot(t, RES['fraction_ever_infected'], 'g-', linewidth=2)
    axes[1,0].set_ylabel('Fraction Ever Infected')
    axes[1,0].set_xlabel('Time Steps')
    axes[1,0].set_title('Cumulative Attack Rate')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Opinion Distribution (final state)
    final_opinions = RES['opinions'][-1, :]
    axes[1,1].hist(final_opinions, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_xlabel('Opinion')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Final Opinion Distribution')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Add parameter info if provided
    if param:
        info_text = f"N_humans={param['n_humans']}, N_bots={param['n_bots']}, Steps={param['N_steps']}"
        fig.text(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    return fig, axes

def plot_multiple_replicas(all_results, param=None, figsize=(12, 8)):
    """
    Plot observables from multiple simulation replicas.
    
    Parameters:
    - all_results: list of RES dictionaries from simulations()
    - param: parameter dictionary (optional)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Multiple Replicas (N={len(all_results)})', fontsize=14)
    
    # Collect data from all replicas
    mean_opinions = np.array([res['mean_opinion'] for res in all_results])
    opinion_vars = np.array([res['opinion_var'] for res in all_results])
    fractions_ever_infected = np.array([res['fraction_ever_infected'] for res in all_results])
    
    t = np.arange(mean_opinions.shape[1])
    
    # Plot 1: Mean Opinion (all replicas + median)
    for i in range(len(all_results)):
        axes[0,0].plot(t, mean_opinions[i], alpha=0.3, color='blue')
    axes[0,0].plot(t, np.median(mean_opinions, axis=0), 'b-', linewidth=3, label='Median')
    axes[0,0].set_ylabel('Mean Opinion')
    axes[0,0].set_title('Average Opinion')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,0].legend()
    
    # Plot 2: Opinion Variance
    for i in range(len(all_results)):
        axes[0,1].plot(t, opinion_vars[i], alpha=0.3, color='red')
    axes[0,1].plot(t, np.median(opinion_vars, axis=0), 'r-', linewidth=3, label='Median')
    axes[0,1].set_ylabel('Opinion Variance')
    axes[0,1].set_title('Opinion Polarization')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Plot 3: Fraction Ever Infected
    for i in range(len(all_results)):
        axes[1,0].plot(t, fractions_ever_infected[i], alpha=0.3, color='green')
    Y = np.median(fractions_ever_infected, axis=0)
    axes[1,0].plot(t, Y, 'g-', linewidth=3, label='Median')
    axes[1,0].set_ylabel('Fraction Ever Infected')
    axes[1,0].set_xlabel('Time Steps')
    axes[1,0].set_title('Cumulative Attack Rate')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    MI = Y[-1]
    
    # Plot 4: Final opinion distributions (all replicas)
    final_opinions_all = np.concatenate([res['opinions'][-1, :] for res in all_results])
    bins = np.linspace(-1, 1, 31)  # Fixed bins from -1 to 1
    axes[1,1].hist(final_opinions_all, bins=bins, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].set_xlabel('Opinion')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Final Opinion Distribution (All Replicas)')
    axes[1,1].set_xlim(-1, 1)  # Fixed x-axis limits
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if param:
        info_text = f"N_humans={param['n_humans']}, N_bots={param['n_bots']}, Replicas={len(all_results)}"
        fig.text(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    return fig, axes, MI



def plot_multiple_replicas_3x3(all_results, param=None, figsize=(15, 12)):
    """
    Plot observables from multiple simulation replicas in a 3x3 grid.
    [0,0]: Mean opinion over time
    [0,1]: Opinion variance over time  
    [0,2] to [2,2]: Individual replica final opinion distributions
    
    Parameters:
    - all_results: list of RES dictionaries from simulations()
    - param: parameter dictionary (optional)
    """
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(f'Multiple Replicas Analysis (N={len(all_results)})', fontsize=16)
    
    # Collect data from all replicas
    mean_opinions = np.array([res['mean_opinion'] for res in all_results])
    opinion_vars = np.array([res['opinion_var'] for res in all_results])
    
    t = np.arange(mean_opinions.shape[1])
    
    # [0,0]: Mean Opinion (all replicas + median)
    for i in range(len(all_results)):
        axes[0,0].plot(t, mean_opinions[i], alpha=0.3, color='blue')
    axes[0,0].plot(t, np.median(mean_opinions, axis=0), 'b-', linewidth=3, label='Median')
    axes[0,0].set_ylabel('Mean Opinion')
    axes[0,0].set_title('Average Opinion Over Time')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,0].legend()
    
    # [0,1]: Opinion Variance
    for i in range(len(all_results)):
        axes[0,1].plot(t, opinion_vars[i], alpha=0.3, color='red')
    axes[0,1].plot(t, np.median(opinion_vars, axis=0), 'r-', linewidth=3, label='Median')
    axes[0,1].set_ylabel('Opinion Variance')
    axes[0,1].set_title('Opinion Polarization Over Time')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # [0,2] to [2,2]: Individual final opinion distributions
    bins = np.linspace(-1, 1, 21)  # Fixed bins from -1 to 1
    
    # Get positions for distribution plots (skip [0,0] and [0,1])
    positions = [(i, j) for i in range(3) for j in range(3) if not (i == 0 and j < 2)]
    
    for idx, (i, j) in enumerate(positions):
        if idx < len(all_results):
            # Plot individual replica distribution
            final_opinions = all_results[idx]['opinions'][-1, :]
            axes[i,j].hist(final_opinions, bins=bins, alpha=0.7, color='purple', edgecolor='black')
            axes[i,j].set_xlabel('Opinion')
            axes[i,j].set_ylabel('Count')
            axes[i,j].set_title(f'Replica {idx+1} Final Distribution')
            axes[i,j].set_xlim(-1, 1)
            axes[i,j].grid(True, alpha=0.3)
            axes[i,j].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        else:
            # Empty subplot for unused positions
            axes[i,j].set_visible(False)
    
    plt.tight_layout()
    
    if param:
        info_text = f"N_humans={param['n_humans']}, N_bots={param['n_bots']}, Replicas={len(all_results)}"
        fig.text(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    return fig, axes


def plot_trajectories_2x4(all_results, param=None, figsize=(8, 16), title=None):
    """
    Plot observables and individual trajectories in a 4x2 grid.
    Row 1: [0,0] Mean opinion, [0,1] Variance
    Row 2: [1,0] Replica 1 trajectories, [1,1] Replica 1 histogram
    Row 3: [2,0] Replica 2 trajectories, [2,1] Replica 2 histogram  
    Row 4: [3,0] Replica 3 trajectories, [3,1] Replica 3 histogram
    
    Parameters:
    - all_results: list of RES dictionaries from simulations()
    - param: parameter dictionary (optional)
    """
    
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    if title is None:
        title = f'Trajectories Analysis (N={len(all_results)})'
    fig.suptitle(title, fontsize=16)
    
    # Collect data from all replicas
    mean_opinions = np.array([res['mean_opinion'] for res in all_results])
    opinion_vars = np.array([res['opinion_var'] for res in all_results])
    
    t = np.arange(mean_opinions.shape[1])
    
    # [0,0]: Mean Opinion (all replicas + median)
    for i in range(len(all_results)):
        axes[0,0].plot(t, mean_opinions[i], alpha=0.3, color='blue')
    axes[0,0].plot(t, np.median(mean_opinions, axis=0), 'b-', linewidth=3, label='Median')
    axes[0,0].set_ylabel('Mean Opinion')
    axes[0,0].set_title('Average Opinion Over Time')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,0].legend()
    
    # [0,1]: Opinion Variance
    for i in range(len(all_results)):
        axes[0,1].plot(t, opinion_vars[i], alpha=0.3, color='red')
    axes[0,1].plot(t, np.median(opinion_vars, axis=0), 'r-', linewidth=3, label='Median')
    axes[0,1].set_ylabel('Opinion Variance')
    axes[0,1].set_title('Opinion Polarization Over Time')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Rows 2-4: First 3 replicas (or fewer if not available)
    n_replicas_to_show = min(3, len(all_results))
    
    for rep_idx in range(n_replicas_to_show):
        row = rep_idx + 1  # Rows 1, 2, 3
        
        # Left side: Individual trajectories
        opinions_over_time = all_results[rep_idx]['opinions']  # Shape: (time_steps, n_humans)
        
        # Plot each human's trajectory
        for human_idx in range(opinions_over_time.shape[1]):
            axes[row, 0].plot(t, opinions_over_time[:, human_idx], alpha=0.3, linewidth=0.5)
        
        axes[row, 0].set_xlabel('Time Steps')
        axes[row, 0].set_ylabel('Opinion')
        axes[row, 0].set_title(f'Replica {rep_idx+1}: All Trajectories')
        axes[row, 0].set_ylim(-1, 1)
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Right side: Final histogram
        final_opinions = all_results[rep_idx]['opinions'][-1, :]
        bins = np.linspace(-1, 1, 21)
        
        axes[row, 1].hist(final_opinions, bins=bins, alpha=0.7, color='purple', edgecolor='black')
        axes[row, 1].set_xlabel('Opinion')
        axes[row, 1].set_ylabel('Count')
        axes[row, 1].set_title(f'Replica {rep_idx+1}: Final Distribution')
        axes[row, 1].set_xlim(-1, 1)
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    # Hide unused rows
    for row in range(n_replicas_to_show + 1, 4):
        axes[row, 0].set_visible(False)
        axes[row, 1].set_visible(False)
    
    plt.tight_layout()
    
    if param:
        info_text = f"N_humans={param['n_humans']}, N_bots={param['n_bots']}, Replicas={len(all_results)}"
        fig.text(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    return fig, axes


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def plot_opinion_space_trajectories(all_results, param=None, figsize=(10, 8)):
    """
    Plot simulation trajectories in (mean_opinion, opinion_variance) space.
    Each trajectory is color-coded from start (blue) to end (red).
    
    Parameters:
    - all_results: list of RES dictionaries from simulations()
    - param: parameter dictionary (optional)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap from blue (start) to red (end)
    colors = ['blue', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('time', colors, N=n_bins)
    
    for i, res in enumerate(all_results):
        mean_opinions = res['mean_opinion']
        opinion_vars = res['opinion_var']
        n_steps = len(mean_opinions)
        
        # Create time-based colors for this trajectory
        time_colors = np.linspace(0, 1, n_steps)
        
        # Plot trajectory as line segments with color gradient
        for j in range(n_steps - 1):
            ax.plot([mean_opinions[j], mean_opinions[j+1]], 
                   [opinion_vars[j], opinion_vars[j+1]], 
                   color=cmap(time_colors[j]), 
                   alpha=0.7, 
                   linewidth=1)
        
        # Mark start and end points
        ax.scatter(mean_opinions[0], opinion_vars[0], 
                  color='blue', s=50, marker='o', alpha=0.8, 
                  label='Start' if i == 0 else "")
        ax.scatter(mean_opinions[-1], opinion_vars[-1], 
                  color='red', s=50, marker='s', alpha=0.8,
                  label='End' if i == 0 else "")
    
    ax.set_xlabel('Mean Opinion')
    ax.set_ylabel('Opinion Variance')
    ax.set_title('System Trajectories in Opinion Space')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar to show time progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Time (normalized)')
    
    if param:
        info_text = f"N_humans={param['n_humans']}, N_bots={param['n_bots']}, Replicas={len(all_results)}"
        fig.text(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax


def plot_opinion_space_simple(all_results, param=None, figsize=(10, 8)):
    """
    Simpler version: each trajectory as a single line with start/end markers.
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, res in enumerate(all_results):
        mean_opinions = res['mean_opinion']
        opinion_vars = res['opinion_var']
        
        # Plot full trajectory
        ax.plot(mean_opinions, opinion_vars, alpha=0.6, linewidth=1.5)
        
        # Mark start (circle) and end (square)
        ax.scatter(mean_opinions[0], opinion_vars[0], 
                  color='green', s=60, marker='o', alpha=0.8)
        ax.scatter(mean_opinions[-1], opinion_vars[-1], 
                  color='red', s=60, marker='s', alpha=0.8)
    
    # Add legend for markers only once
    ax.scatter([], [], color='green', s=60, marker='o', label='Start')
    ax.scatter([], [], color='red', s=60, marker='s', label='End')
    
    ax.set_xlabel('Mean Opinion')
    ax.set_ylabel('Opinion Variance')
    ax.set_title('System Trajectories in Opinion Space')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if param:
        info_text = f"N_humans={param['n_humans']}, N_bots={param['n_bots']}, Replicas={len(all_results)}"
        fig.text(0.02, 0.02, info_text, fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    return fig, ax