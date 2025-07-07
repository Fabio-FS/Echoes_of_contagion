import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt

def generate_parameter_grid():
    """Same grid as in your grid_search.py"""
    comm_errors = [0, 0.05, 0.1, 0.2]
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    return list(itertools.product(comm_errors, thresholds, n_bots))

def check_results_status(results_dir="."):
    """Check which simulation runs completed by reading pickle filenames"""
    combinations = generate_parameter_grid()
    total_runs = len(combinations)
    
    # Find all pickle files (any pattern that looks like results)
    patterns = [
        "results_ce*_th*_nb*_*.pkl",
        "results_*.pkl", 
        "*.pkl"
    ]
    
    existing_files = []
    for pattern in patterns:
        files = glob.glob(os.path.join(results_dir, pattern))
        existing_files.extend(files)
        if files:
            break  # Use first pattern that finds files
    
    print(f"Found {len(existing_files)} pickle files")
    if existing_files:
        print(f"Example filename: {os.path.basename(existing_files[0])}")
    
    # Parse existing files to get parameters
    completed_runs = set()
    failed_parses = []
    
    for filepath in existing_files:
        filename = os.path.basename(filepath)
        
        # Try multiple parsing strategies
        try:
            # Strategy 1: results_ce0.05_th-0.4_nb20_2025_07_04.pkl
            if "results_ce" in filename:
                parts = filename.replace("results_", "").split("_")
                ce = float(parts[0].replace("ce", ""))
                th = float(parts[1].replace("th", ""))
                nb = int(parts[2].replace("nb", ""))
                completed_runs.add((ce, th, nb))
            # Strategy 2: other naming patterns...
            else:
                failed_parses.append(filename)
                
        except Exception as e:
            failed_parses.append(filename)
    
    if failed_parses:
        print(f"\nCouldn't parse {len(failed_parses)} filenames:")
        for f in failed_parses[:5]:
            print(f"  {f}")
    
    # Map completed runs back to job IDs
    missing_runs = []
    completed_jobs = []
    
    for job_id, (ce, th, nb) in enumerate(combinations):
        if (ce, th, nb) in completed_runs:
            completed_jobs.append(job_id)
        else:
            missing_runs.append((job_id, ce, th, nb))
    
    print(f"\nTotal expected runs: {total_runs}")
    print(f"Completed runs: {len(completed_runs)}")
    print(f"Missing runs: {len(missing_runs)}")
    print(f"Success rate: {len(completed_runs)/total_runs*100:.1f}%")
    
    if missing_runs:
        print(f"\nMissing job IDs: {[job_id for job_id, _, _, _ in missing_runs]}")
        print("\nFirst 10 missing runs:")
        for job_id, ce, th, nb in missing_runs[:10]:
            print(f"  Job {job_id}: ce={ce}, th={th}, nb={nb}")
        if len(missing_runs) > 10:
            print(f"  ... and {len(missing_runs)-10} more")
    
    return completed_jobs, missing_runs

def plot_completion_map(results_dir="."):
    """Create a visual map of completed vs missing runs"""
    combinations = generate_parameter_grid()
    completed_jobs, missing_runs = check_results_status(results_dir)
    
    # Create completion matrix
    comm_errors = [0, 0.05, 0.1, 0.2]
    thresholds = [-0.8, -0.4, 0, +0.4, 0.8]
    n_bots = [0, 10, 20, 40, 80]
    
    # For each ce value, create a 2D map of th vs nb
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    for i, ce in enumerate(comm_errors):
        completion_matrix = np.zeros((len(thresholds), len(n_bots)))
        
        for job_id, (ce_combo, th_combo, nb_combo) in enumerate(combinations):
            if ce_combo == ce:
                th_idx = thresholds.index(th_combo)
                nb_idx = n_bots.index(nb_combo)
                completion_matrix[th_idx, nb_idx] = 1 if job_id in completed_jobs else 0
        
        im = axes[i].imshow(completion_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[i].set_title(f'CE={ce}')
        axes[i].set_xlabel('N_bots')
        axes[i].set_ylabel('Threshold')
        axes[i].set_xticks(range(len(n_bots)))
        axes[i].set_xticklabels(n_bots)
        axes[i].set_yticks(range(len(thresholds)))
        axes[i].set_yticklabels(thresholds)
        
        # Add text annotations
        for th_idx in range(len(thresholds)):
            for nb_idx in range(len(n_bots)):
                text = '✓' if completion_matrix[th_idx, nb_idx] == 1 else '✗'
                color = 'white' if completion_matrix[th_idx, nb_idx] == 1 else 'black'
                axes[i].text(nb_idx, th_idx, text, ha='center', va='center', color=color, fontsize=12)
    
    plt.tight_layout()
    plt.suptitle(f'Simulation Completion Map (Green=Complete, Red=Missing)', y=1.02)
    plt.show()
    
    return completed_jobs, missing_runs

def analyze_results(results_dir=".", show_plot=True):
    """Main function to analyze simulation results"""
    if show_plot:
        completed_jobs, missing_runs = plot_completion_map(results_dir)
    else:
        completed_jobs, missing_runs = check_results_status(results_dir)
    
    if missing_runs:
        print(f"\nTo rerun missing jobs, use these job IDs:")
        missing_job_ids = [job_id for job_id, _, _, _ in missing_runs]
        print(f"sbatch --array={','.join(map(str, missing_job_ids))} submit_grid.sh")
    
    return completed_jobs, missing_runs

# Run the analysis if called directly
if __name__ == "__main__":
    analyze_results()