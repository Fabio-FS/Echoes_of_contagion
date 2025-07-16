import numpy as np

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


def get_simple_frequencies(consolidated_results):
    """One-liner to get the frequencies"""
    results = classify_all_replicas_simple(consolidated_results)
    print_simple_summary(results)
    return results['state_fractions']


# Usage:
# frequencies = get_simple_frequencies(your_consolidated_results)