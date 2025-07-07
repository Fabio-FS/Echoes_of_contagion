#!/usr/bin/env python3
"""
Setup script to create directory structure for a new simulation run.
Run this BEFORE submitting the job array with sbatch.
"""

import os
import datetime
import glob

def get_next_run_id():
    """Automatically determine the next run ID for today's date"""
    today = datetime.date.today().strftime("%Y_%m_%d")
    
    # Look for existing folders with today's date
    pattern = f"RESULTS/{today}_*"
    existing_folders = glob.glob(pattern)
    
    if not existing_folders:
        # No folders for today, start with 1
        run_id = 1
    else:
        # Extract run numbers from existing folders
        run_numbers = []
        for folder in existing_folders:
            try:
                # Extract number after last underscore
                run_num = int(folder.split('_')[-1])
                run_numbers.append(run_num)
            except ValueError:
                # Skip folders that don't end with a number
                continue
        
        # Get next available number
        if run_numbers:
            run_id = max(run_numbers) + 1
        else:
            run_id = 1
    
    return today, run_id

def create_run_directories():
    """Create the directory structure for this run"""
    date_str, run_id = get_next_run_id()
    
    base_dir = f"RESULTS/{date_str}_{run_id}"
    pickle_dir = f"{base_dir}/pickles"
    log_dir = f"{base_dir}/logs"
    
    # Create directories
    os.makedirs(pickle_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Created run directory: {base_dir}")
    print(f"Pickles will be saved to: {pickle_dir}")
    print(f"Logs will be saved to: {log_dir}")
    
    return base_dir, pickle_dir, log_dir

def main():
    """Main function to set up the run"""
    print("Setting up new simulation run...")
    
    # Create directory structure
    base_dir, pickle_dir, log_dir = create_run_directories()
    
    # Save the paths to a file that grid_search.py can read
    with open("current_run_dirs.txt", "w") as f:
        f.write(f"{pickle_dir}\n{log_dir}\n{base_dir}")
    
    print(f"\nDirectory paths saved to: current_run_dirs.txt")
    print(f"You can now submit your job array with: sbatch submit_grid.sh")

if __name__ == "__main__":
    main()