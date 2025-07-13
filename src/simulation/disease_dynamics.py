import numpy as np
import igraph as ig
from numba import jit

@jit(nopython=True)
def update_susceptibilities_jit_discrete(opinions, beta0, n_groups=5):
    """
    Vectorized JIT-compiled susceptibility calculation with discrete behavioral groups
    
    Args:
        opinions: array of opinion values in [-1, 1]
        beta0: maximum susceptibility 
        n_groups: number of discrete behavioral groups (default 5)
    
    Returns:
        susceptibilities: array where each agent belongs to one of n_groups
                         with linearly spaced susceptibility values
    """
    # Create n_groups equally spaced susceptibility levels from beta0 to 0
    # Group 0 (most anti-mask): susceptibility = beta0  
    # Group n_groups-1 (most pro-mask): susceptibility = 0
    susceptibility_levels = np.linspace(beta0, 0, n_groups)
    
    # Vectorized group assignment using digitize
    # Map opinions from [-1, 1] to group indices [0, n_groups-1]
    # Note: digitize returns indices 1-based, so we subtract 1
    group_indices = np.digitize(opinions, np.linspace(-1, 1, n_groups + 1)) - 1
    
    # Clip to valid range [0, n_groups-1] to handle edge cases
    group_indices = np.clip(group_indices, 0, n_groups - 1)
    
    # Vectorized lookup of susceptibilities
    susceptibilities = susceptibility_levels[group_indices]
    
    return susceptibilities


def update_susceptibilities_discrete(graph, n_groups=5):
    """
    Update susceptibilities using discrete behavioral groups
    
    Args:
        graph: igraph object
        n_groups: number of discrete groups (default 5)
                 2 -> [-1,0), [0,1] 
                 5 -> [-1,-0.6), [-0.6,-0.2), [-0.2,0.2), [0.2,0.6), [0.6,1]
    """
    beta0 = graph["beta0"]
    
    # Get all human opinions as numpy array
    opinions = np.array(graph.vs['opinion'][:graph["n_humans"]])

    # Use JIT-compiled function with discrete groups
    susceptibilities = update_susceptibilities_jit_discrete(opinions, beta0, n_groups)
    
    # Pad with zeros for bots and store back to graph
    all_susceptibilities = list(susceptibilities) + [0] * graph["n_bots"]
    graph.vs['susceptibility'] = all_susceptibilities


@jit(nopython=True)
def update_susceptibilities_jit(opinions, beta0):
    """JIT-compiled susceptibility calculation"""
    # behavior represent the fraction of time spent wearing a "perfect" mask that sets beta0 to 0.
    susceptibilities = beta0 * (1 - (opinions + 1)/2)   # Vectorized susceptibility calculation
    return susceptibilities

def disease_dynamic_step(graph):
    #update_susceptibilities(graph)
    update_susceptibilities_discrete(graph)
    run_SIR_step(graph)

def update_susceptibilities(graph):
    beta0 = graph["beta0"]

    
    # Get all human opinions as numpy array
    opinions = np.array(graph.vs['opinion'])  # I also take the opinion of bots. To use numpy vectorization, but their susceptibilities will never be used.

    # Use JIT-compiled function
    susceptibilities = update_susceptibilities_jit(opinions, beta0)
    
    # Store back to graph - THIS WAS MISSING!
    graph.vs['susceptibility'] = susceptibilities

def run_SIR_step(graph):
    health_states = np.array(graph.vs['health_state'][:graph["n_humans"]])
    if np.sum(health_states == 1) == 0:
        return  # No infected, skip entirely
    else:
        randoms = np.random.rand(graph["n_humans"])
        recovery_rate = graph["recovery_rate"]
        
        # Get states and susceptibilities as arrays
        susceptibilities = np.array(graph.vs['susceptibility'][:graph["n_humans"]])
        #if np.random.rand() < 0.01:  # Print ~1% of the time
        #    print(f"susceptibility range = {np.min(susceptibilities):.6f} to {np.max(susceptibilities):.6f}")
        
        # Vectorized recovery
        infected_mask = (health_states == 1)
        
        new_recoveries = np.where(infected_mask & (randoms < recovery_rate))[0]
        
        # Infection step - still needs loop for neighbor counting
        susceptible_mask = (health_states == 0)
        
        n_infected_neighbors = np.zeros(graph["n_humans"])
        for i in range(graph["n_humans"]):
            if susceptible_mask[i] == True:
                n_infected_neighbors[i] = np.sum(health_states[graph.vs[i]['human_neighbors']] == 1)

        new_infections = np.where(susceptible_mask & (randoms < 1- np.power((1-susceptibilities), n_infected_neighbors)))[0]

        # Apply changes
        health_states[new_recoveries] = 2
        health_states[new_infections] = 1
        graph.vs["health_state"] = list(health_states) + [-1] * graph["n_bots"]  # list

