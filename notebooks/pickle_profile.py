import pickle
import numpy as np
import sys

def profile_pickle_memory(filepath):
    """Profile memory usage of pickle file components"""
    
    # Load the data
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total file size: {get_file_size(filepath):.2f} MB")
    print("\n" + "="*50)
    print("MEMORY BREAKDOWN:")
    print("="*50)
    
    # Profile top-level components
    total_size = 0
    for key, value in data.items():
        size_mb = get_object_size(value) / (1024**2)
        total_size += size_mb
        print(f"{key:20}: {size_mb:8.2f} MB")
    
    print(f"{'Total in memory':20}: {total_size:8.2f} MB")
    
    # Detailed breakdown of results
    if 'all_results' in data:
        print(f"\n{'RESULTS BREAKDOWN':20}")
        print("-"*30)
        
        results = data['all_results']
        n_replicas = len(results)
        print(f"Number of replicas: {n_replicas}")
        
        if n_replicas > 0:
            # Analyze first replica in detail
            replica = results[0]
            replica_size = get_object_size(replica) / (1024**2)
            print(f"Size per replica: {replica_size:.2f} MB")
            print(f"Projected 100 replicas: {replica_size * 100:.2f} MB")
            
            print(f"\nPer-replica component sizes:")
            for key, value in replica.items():
                if isinstance(value, np.ndarray):
                    size_kb = value.nbytes / 1024
                    print(f"  {key:20}: {size_kb:8.1f} KB  {value.shape} {value.dtype}")
                else:
                    size_kb = get_object_size(value) / 1024
                    print(f"  {key:20}: {size_kb:8.1f} KB  {type(value).__name__}")

def get_file_size(filepath):
    """Get file size in MB"""
    import os
    return os.path.getsize(filepath) / (1024**2)

def get_object_size(obj):
    """Get object size in bytes using pickle serialization"""
    return len(pickle.dumps(obj))

def suggest_optimizations(replica_data):
    """Suggest memory optimizations based on data analysis"""
    print(f"\n{'OPTIMIZATION SUGGESTIONS':20}")
    print("-"*40)
    
    total_savings = 0
    
    for key, value in replica_data.items():
        if isinstance(value, np.ndarray):
            current_size = value.nbytes
            
            # Check if we can use smaller dtypes
            if value.dtype == np.float64:
                savings = current_size / 2  # float32 is half the size
                total_savings += savings
                print(f"• {key}: Convert float64→float32 saves {savings/1024:.1f} KB")
            
            elif value.dtype == np.int64:
                if np.max(value) < 32767:  # fits in int16
                    savings = current_size * 3/4  # int16 is 1/4 the size
                    total_savings += savings
                    print(f"• {key}: Convert int64→int16 saves {savings/1024:.1f} KB")
                elif np.max(value) < 2147483647:  # fits in int32
                    savings = current_size / 2  # int32 is half the size
                    total_savings += savings
                    print(f"• {key}: Convert int64→int32 saves {savings/1024:.1f} KB")
    
    print(f"\nTotal potential savings per replica: {total_savings/1024:.1f} KB")
    print(f"Savings for 100 replicas: {total_savings*100/(1024**2):.1f} MB")

# Example usage
if __name__ == "__main__":
    # Replace with your actual pickle file path
    filepath = "results/your_simulation_results.pkl"
    
    try:
        profile_pickle_memory(filepath)
        
        # Load just one replica for optimization analysis
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        if data['all_results']:
            suggest_optimizations(data['all_results'][0])
            
    except FileNotFoundError:
        print(f"File {filepath} not found. Please update the filepath.")
        print("\nAlternatively, you can run this on a single replica:")
        print("replica_data = {'opinions': np.random.rand(200, 1000).astype(np.float64), ...}")
        print("suggest_optimizations(replica_data)")