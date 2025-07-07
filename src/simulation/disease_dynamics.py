import numpy as np
import igraph as ig
from numba import jit



@jit(nopython=True)
def update_susceptibilities_jit(opinions, beta0, O0, behavior_strength):
    """JIT-compiled susceptibility calculation"""
    # behavior represent the fraction of time spent wearing a "perfect" mask that sets beta0 to 0.
    behavior = 1 / (1 + np.exp(-behavior_strength * (opinions - O0))) # Vectorized logistic function
    susceptibilities = beta0 * (1 - behavior) # Vectorized susceptibility calculation
    return susceptibilities

def disease_dynamic_step(graph):
    update_susceptibilities(graph)
    run_SIR_step(graph)

def update_susceptibilities(graph):
    beta0 = graph["beta0"]
    O0 = graph["O0"]
    behavior_strength = graph["behavior_strength"]
    
    # Get all human opinions as numpy array
    opinions = np.array(graph.vs['opinion'])  # I also take the opinion of bots. To use numpy vectorization, but their susceptibilities will never be used.

    # Use JIT-compiled function
    susceptibilities = update_susceptibilities_jit(opinions, beta0, O0, behavior_strength)
    
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

