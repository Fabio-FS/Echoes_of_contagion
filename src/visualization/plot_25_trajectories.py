import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

def load_simulation_data(results_dir, ce_value):
    """Load all simulation results for a specific communication error value"""
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    data_grid = {}
    
    for th in thresholds:
        for nb in n_bots:
            # Find the corresponding file
            pattern = f"results_ce{ce_value}_th{th}_nb{nb}_*.pkl"
            files = glob.glob(os.path.join(results_dir, pattern))
            
            if files:
                try:
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                    data_grid[(th, nb)] = data
                    print(f"Loaded: th={th}, nb={nb}")
                except Exception as e:
                    print(f"Error loading th={th}, nb={nb}: {e}")
                    data_grid[(th, nb)] = None
            else:
                print(f"Missing: th={th}, nb={nb}")
                data_grid[(th, nb)] = None
    
    return data_grid




import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import os

def plot_opinion_space_grid(results_dir, ce_value=0, figsize=(15, 15), 
                           xlim=None, ylim=None):
    """Create 5x5 grid of opinion space trajectory plots (mean vs variance)
    Loads one file at a time to save memory"""
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    # Create the plot
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    fig.suptitle(f'Opinion Space Trajectories Grid (Communication Error = {ce_value})', fontsize=16)
    
    for i, th in enumerate(thresholds):
        for j, nb in enumerate(n_bots):
            ax = axes[i, j]
            
            # Find and load the specific file for this parameter combination
            pattern = f"results_ce{ce_value}_th{th}_nb{nb}_*.pkl"
            files = glob.glob(os.path.join(results_dir, pattern))
            
            if files:
                try:
                    print(f"Loading th={th}, nb={nb}...")
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                    
                    all_results = data['all_results']
                    
                    # Plot trajectories in opinion space for each replica
                    for replica in all_results:
                        mean_opinions = replica['mean_opinion']
                        opinion_vars = replica['opinion_var']
                        
                        # Plot trajectory line
                        ax.plot(mean_opinions, opinion_vars, alpha=0.2, color='purple', linewidth=1)
                        
                        # Mark start (circle) and end (square)
                        ax.scatter(mean_opinions[0], opinion_vars[0], 
                                  color='green', s=20, marker='o', alpha=0.8)
                        ax.scatter(mean_opinions[-1], opinion_vars[-1], 
                                  color='red', s=20, marker='s', alpha=0.8)
                    
                    ax.set_xlim(xlim if xlim else (-1, 1))
                    ax.set_ylim(ylim if ylim else (0, 1))
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                    
                    # Clear data from memory immediately
                    del data
                    del all_results
                    
                except Exception as e:
                    print(f"Error loading th={th}, nb={nb}: {e}")
                    ax.text(0.5, 0.5, 'Load\nError', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='red')
                    ax.set_xlim(xlim if xlim else (-1, 1))
                    ax.set_ylim(ylim if ylim else (0, 1))
            else:
                print(f"Missing: th={th}, nb={nb}")
                ax.text(0.5, 0.5, 'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(xlim if xlim else (-1, 1))
                ax.set_ylim(ylim if ylim else (0, 1))
            
            # Labels and titles
            if i == 0:
                ax.set_title(f'n_bots={nb}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'th={th}', fontsize=10)
            if i == len(thresholds) - 1:
                ax.set_xlabel('Mean Opinion', fontsize=8)
            if j == 0 and i == len(thresholds) - 1:
                ax.set_ylabel('Opinion Variance', fontsize=8)
            
            # Remove tick labels except for edges
            if i < len(thresholds) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    
    # Add legend in the last subplot
    axes[0, -1].scatter([], [], color='green', s=40, marker='o', label='Start')
    axes[0, -1].scatter([], [], color='red', s=40, marker='s', label='End')
    axes[0, -1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig, axes

def plot_final_opinion_distribution_grid(results_dir, ce_value=0, figsize=(15, 15), 
                                        xlim=None, bins=20):
    """Create 5x5 grid of final opinion distribution histograms
    Loads one file at a time to save memory"""
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    # Create the plot
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    fig.suptitle(f'Final Opinion Distributions Grid (Communication Error = {ce_value})', fontsize=16)
    
    for i, th in enumerate(thresholds):
        for j, nb in enumerate(n_bots):
            ax = axes[i, j]
            
            # Find and load the specific file for this parameter combination
            pattern = f"results_ce{ce_value}_th{th}_nb{nb}_*.pkl"
            files = glob.glob(os.path.join(results_dir, pattern))
            
            if files:
                try:
                    print(f"Loading th={th}, nb={nb}...")
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                    
                    all_results = data['all_results']
                    
                    # Collect final opinions from all replicas
                    final_opinions_all = []
                    for replica in all_results:
                        final_opinions = replica['opinions'][-1, :]  # Last time step, all humans
                        final_opinions_all.extend(final_opinions)
                    
                    # Plot normalized histogram (fractions that sum to 1)
                    bin_range = xlim if xlim else (-1, 1)
                    weights = np.ones_like(final_opinions_all) / len(final_opinions_all)
                    ax.hist(final_opinions_all, bins=bins, alpha=0.7, color='purple', 
                           edgecolor='black', range=bin_range, weights=weights)
                    ax.set_xlim(bin_range)
                    ax.set_ylim(0, 0.5)
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                    
                    # Calculate percentages below and above 0
                    below_zero = np.sum(np.array(final_opinions_all) < 0) / len(final_opinions_all) * 100
                    above_zero = np.sum(np.array(final_opinions_all) > 0) / len(final_opinions_all) * 100
                    
                    # Add percentage text (left side for <0, right side for >0)
                    ax.text(0.05, 0.95, f'{below_zero:.0f}%', transform=ax.transAxes, 
                           fontsize=8, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='lightblue', alpha=0.7))
                    ax.text(0.95, 0.95, f'{above_zero:.0f}%', transform=ax.transAxes, 
                           fontsize=8, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='lightcoral', alpha=0.7))
                    
                    # Clear data from memory immediately
                    del data
                    del all_results
                    del final_opinions_all
                    
                except Exception as e:
                    print(f"Error loading th={th}, nb={nb}: {e}")
                    ax.text(0.5, 0.5, 'Load\nError', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='red')
                    ax.set_xlim(xlim if xlim else (-1, 1))
                    ax.set_ylim(0, 0.5)
            else:
                print(f"Missing: th={th}, nb={nb}")
                ax.text(0.5, 0.5, 'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(xlim if xlim else (-1, 1))
            
            # Labels and titles
            if i == 0:
                ax.set_title(f'n_bots={nb}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'th={th}', fontsize=10)
            if i == len(thresholds) - 1:
                ax.set_xlabel('Final Opinion', fontsize=8)
            if j == 0 and i == len(thresholds) - 1:
                ax.set_ylabel('Fraction', fontsize=8)
            
            # Remove tick labels except for edges
            if i < len(thresholds) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    
    plt.tight_layout()
    return fig, axes

def plot_opinion_space_grid(results_dir, ce_value=0, figsize=(15, 15), 
                           xlim=None, ylim=None):
    """Create 5x5 grid of opinion space trajectory plots (mean vs variance)
    Loads one file at a time to save memory"""
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    # Create the plot
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    fig.suptitle(f'Opinion Space Trajectories Grid (Communication Error = {ce_value})', fontsize=16)
    
    for i, th in enumerate(thresholds):
        for j, nb in enumerate(n_bots):
            ax = axes[i, j]
            
            # Find and load the specific file for this parameter combination
            pattern = f"results_ce{ce_value}_th{th}_nb{nb}_*.pkl"
            files = glob.glob(os.path.join(results_dir, pattern))
            
            if files:
                try:
                    print(f"Loading th={th}, nb={nb}...")
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                    
                    all_results = data['all_results']
                    
                    # Plot trajectories in opinion space for each replica
                    for replica in all_results:
                        mean_opinions = replica['mean_opinion']
                        opinion_vars = replica['opinion_var']
                        
                        # Plot trajectory line
                        ax.plot(mean_opinions, opinion_vars, alpha=0.6, linewidth=1)
                        
                        # Mark start (circle) and end (square)
                        ax.scatter(mean_opinions[0], opinion_vars[0], 
                                  color='green', s=20, marker='o', alpha=0.8)
                        ax.scatter(mean_opinions[-1], opinion_vars[-1], 
                                  color='red', s=20, marker='s', alpha=0.8)
                    
                    ax.set_xlim(xlim if xlim else (-1, 1))
                    ax.set_ylim(ylim if ylim else (0, 1))
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                    
                    # Clear data from memory immediately
                    del data
                    del all_results
                    
                except Exception as e:
                    print(f"Error loading th={th}, nb={nb}: {e}")
                    ax.text(0.5, 0.5, 'Load\nError', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='red')
                    ax.set_xlim(xlim if xlim else (-1, 1))
                    ax.set_ylim(ylim if ylim else (0, 1))
            else:
                print(f"Missing: th={th}, nb={nb}")
                ax.text(0.5, 0.5, 'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(xlim if xlim else (-1, 1))
                ax.set_ylim(ylim if ylim else (0, 1))
            
            # Labels and titles
            if i == 0:
                ax.set_title(f'n_bots={nb}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'th={th}', fontsize=10)
            if i == len(thresholds) - 1:
                ax.set_xlabel('Mean Opinion', fontsize=8)
            if j == 0 and i == len(thresholds) - 1:
                ax.set_ylabel('Opinion Variance', fontsize=8)
            
            # Remove tick labels except for edges
            if i < len(thresholds) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    
    # Add legend in the last subplot
    axes[0, -1].scatter([], [], color='green', s=40, marker='o', label='Start')
    axes[0, -1].scatter([], [], color='red', s=40, marker='s', label='End')
    axes[0, -1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    return fig, axes



def plot_final_opinion_distribution_grid_with_insets(results_dir, ce_value=0, figsize=(15, 15), 
                                                   xlim=None, bins=20, inset_xlim=None, inset_ylim=None):
    """Create 5x5 grid of final opinion distribution histograms with trajectory insets
    Loads one file at a time to save memory"""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    # Create the plot
    fig, axes = plt.subplots(5, 5, figsize=figsize)
    fig.suptitle(f'Final Opinion Distributions + Trajectories (Communication Error = {ce_value})', fontsize=16)
    
    for i, th in enumerate(thresholds):
        for j, nb in enumerate(n_bots):
            ax = axes[i, j]
            
            # Find and load the specific file for this parameter combination
            pattern = f"results_ce{ce_value}_th{th}_nb{nb}_*.pkl"
            files = glob.glob(os.path.join(results_dir, pattern))
            
            if files:
                try:
                    print(f"Loading th={th}, nb={nb}...")
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                    
                    all_results = data['all_results']
                    
                    # Collect final opinions from all replicas
                    final_opinions_all = []
                    for replica in all_results:
                        final_opinions = replica['opinions'][-1, :]  # Last time step, all humans
                        final_opinions_all.extend(final_opinions)
                    
                    # Plot normalized histogram (fractions that sum to 1)
                    bin_range = xlim if xlim else (-1, 1)
                    weights = np.ones_like(final_opinions_all) / len(final_opinions_all)
                    ax.hist(final_opinions_all, bins=bins, alpha=0.7, color='purple', 
                           edgecolor='black', range=bin_range, weights=weights)
                    ax.set_xlim(bin_range)
                    ax.set_ylim(0, 0.5)
                    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                    
                    # Calculate percentages below and above 0
                    below_zero = np.sum(np.array(final_opinions_all) < 0) / len(final_opinions_all) * 100
                    above_zero = np.sum(np.array(final_opinions_all) > 0) / len(final_opinions_all) * 100
                    
                    # Add percentage text (red for <0, blue for >0)
                    ax.text(0.05, 0.95, f'{below_zero:.0f}%', transform=ax.transAxes, 
                           fontsize=8, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='lightcoral', alpha=0.7))
                    ax.text(0.95, 0.95, f'{above_zero:.0f}%', transform=ax.transAxes, 
                           fontsize=8, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.2', 
                           facecolor='lightblue', alpha=0.7))
                    
                    # Create inset for trajectory plot
                    inset = inset_axes(ax, width="40%", height="35%", loc='upper right', 
                                     bbox_to_anchor=(0, -0.15, 1, 1), bbox_transform=ax.transAxes)
                    
                    # Plot trajectories in the inset
                    for replica in all_results:
                        mean_opinions = replica['mean_opinion']
                        opinion_vars = replica['opinion_var']
                        
                        # Plot trajectory line
                        inset.plot(mean_opinions, opinion_vars, alpha=0.4, linewidth=0.5, color='purple')
                        
                        # Mark start (circle) and end (square) - smaller markers
                        inset.scatter(mean_opinions[0], opinion_vars[0], 
                                    color='green', s=3, marker='o', alpha=0.6)
                        inset.scatter(mean_opinions[-1], opinion_vars[-1], 
                                    color='red', s=3, marker='s', alpha=0.6)
                    
                    # Set inset limits
                    inset.set_xlim(inset_xlim if inset_xlim else (-1, 1))
                    inset.set_ylim(inset_ylim if inset_ylim else (0, 1))
                    inset.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
                    
                    # Remove inset ticks and labels
                    inset.set_xticks([])
                    inset.set_yticks([])
                    
                    # Clear data from memory immediately
                    del data
                    del all_results
                    del final_opinions_all
                    
                except Exception as e:
                    print(f"Error loading th={th}, nb={nb}: {e}")
                    ax.text(0.5, 0.5, 'Load\nError', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, color='red')
                    ax.set_xlim(xlim if xlim else (-1, 1))
                    ax.set_ylim(0, 0.5)
            else:
                print(f"Missing: th={th}, nb={nb}")
                ax.text(0.5, 0.5, 'Missing\nData', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
                ax.set_xlim(xlim if xlim else (-1, 1))
                ax.set_ylim(0, 0.5)
            
            # Labels and titles
            if i == 0:
                ax.set_title(f'n_bots={nb}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'th={th}', fontsize=10)
            if i == len(thresholds) - 1:
                ax.set_xlabel('Final Opinion', fontsize=8)
            if j == 0 and i == len(thresholds) - 1:
                ax.set_ylabel('Fraction', fontsize=8)
            
            # Remove tick labels except for edges
            if i < len(thresholds) - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    
    plt.tight_layout()
    return fig, axes


