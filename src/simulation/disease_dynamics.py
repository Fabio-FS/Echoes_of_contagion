import numpy as np
import igraph as ig
from numba import jit

# ==================== VACCINATION PROBABILITY FUNCTIONS ====================

@jit(nopython=True)
def update_vaccination_probabilities_jit(opinions, xi_max):
    """
    Opinion correlates with vaccination probability: -1 → 0, +1 → xi_max
    
    Args:
        opinions: array of opinion values in [-1, 1]
        xi_max: maximum vaccination probability
    
    Returns:
        vaccination_probs: array where each agent has vaccination probability
    """
    # Linear mapping: opinion -1 → prob 0, opinion +1 → prob xi_max
    vaccination_probs = xi_max * (opinions + 1) / 2
    return vaccination_probs

@jit(nopython=True)
def update_vaccination_probabilities_jit_discrete(opinions, xi_max, n_groups=5):
    """
    Discrete vaccination probability groups (similar to discrete susceptibility)
    
    Args:
        opinions: array of opinion values in [-1, 1]
        xi_max: maximum vaccination probability
        n_groups: number of discrete behavioral groups
    
    Returns:
        vaccination_probs: array where each agent belongs to discrete group
    """
    # Create n_groups equally spaced vaccination levels from 0 to xi_max
    # Group 0 (most anti-vaccine): vaccination_prob = 0
    # Group n_groups-1 (most pro-vaccine): vaccination_prob = xi_max
    vaccination_levels = np.linspace(0, xi_max, n_groups)
    
    # Map opinions to group indices
    group_indices = np.digitize(opinions, np.linspace(-1, 1, n_groups + 1)) - 1
    group_indices = np.clip(group_indices, 0, n_groups - 1)
    
    # Vectorized lookup
    vaccination_probs = vaccination_levels[group_indices]
    return vaccination_probs

def update_vaccination_probabilities(graph):
    """Update vaccination probabilities based on opinions"""
    xi_max = graph["xi_max"]
    
    # Get human opinions
    opinions = np.array(graph.vs['opinion'][:graph["n_humans"]])
    
    # Choose discrete or continuous (matching your susceptibility approach)
    use_discrete = graph["use_discrete_vaccination"] if "use_discrete_vaccination" in graph.attributes() else True
    if use_discrete:
        n_groups = graph["vaccination_groups"] if "vaccination_groups" in graph.attributes() else 5
        vaccination_probs = update_vaccination_probabilities_jit_discrete(opinions, xi_max, n_groups)
    else:
        vaccination_probs = update_vaccination_probabilities_jit(opinions, xi_max)
    
    # Pad with zeros for bots and store
    all_vaccination_probs = list(vaccination_probs) + [0] * graph["n_bots"]
    graph.vs['vaccination_probability'] = all_vaccination_probs

# ==================== SIRV DISEASE DYNAMICS ====================

def disease_dynamic_step_SIRV(graph):
    """Main SIRV disease step"""
    update_vaccination_probabilities(graph)
    run_SIRV_step(graph)

def run_SIRV_step(graph):
    """SIRV step with vaccination, infection, and recovery"""
    health_states = np.array(graph.vs['health_state'][:graph["n_humans"]])
    
    # Early exit if no infected
    if np.sum(health_states == 1) == 0:
        return
    
    randoms = np.random.rand(graph["n_humans"])
    recovery_rate = graph["recovery_rate"]
    
    # Get vaccination probabilities
    vaccination_probs = np.array(graph.vs['vaccination_probability'][:graph["n_humans"]])
    
    # ==================== VACCINATION STEP ====================
    # Only susceptible people can get vaccinated (S → V)
    susceptible_mask = (health_states == 0)
    
    # Apply vaccination probability per person per timestep
    new_vaccinations = np.where(susceptible_mask & (randoms < vaccination_probs))[0]
    
    # ==================== RECOVERY STEP ====================
    # Infected people recover (I → R)
    infected_mask = (health_states == 1)
    new_recoveries = np.where(infected_mask & (randoms < recovery_rate))[0]
    
    # ==================== INFECTION STEP ====================
    # Only susceptible people can get infected (S → I)
    # Vaccinated people are immune
    
    # Count infected neighbors for susceptible people
    n_infected_neighbors = np.zeros(graph["n_humans"])
    for i in range(graph["n_humans"]):
        if susceptible_mask[i]:
            n_infected_neighbors[i] = np.sum(health_states[graph.vs[i]['human_neighbors']] == 1)
    
    # Use base susceptibility (beta0) for all susceptible people
    # In SIRV, behavior affects vaccination, not susceptibility during infection
    beta0 = graph["beta0"]
    infection_prob = 1 - np.power((1 - beta0), n_infected_neighbors)
    
    new_infections = np.where(susceptible_mask & (randoms < infection_prob))[0]
    
    # ==================== APPLY STATE CHANGES ====================
    # Apply in order: vaccination, recovery, infection
    # Note: Use separate random numbers to avoid conflicts
    vaccination_randoms = np.random.rand(graph["n_humans"])
    health_states[susceptible_mask & (vaccination_randoms < vaccination_probs)] = 3  # V
    
    # Update masks after vaccination
    health_states = np.array(graph.vs['health_state'][:graph["n_humans"]])  # Refresh
    infected_mask = (health_states == 1)
    susceptible_mask = (health_states == 0)  # Update after vaccinations
    
    recovery_randoms = np.random.rand(graph["n_humans"])
    health_states[infected_mask & (recovery_randoms < recovery_rate)] = 2  # R
    
    # Update again after recovery
    health_states = np.array(graph.vs['health_state'][:graph["n_humans"]])
    susceptible_mask = (health_states == 0)
    
    infection_randoms = np.random.rand(graph["n_humans"])
    for i in range(graph["n_humans"]):
        if susceptible_mask[i]:
            n_infected = np.sum(health_states[graph.vs[i]['human_neighbors']] == 1)
            if infection_randoms[i] < 1 - np.power((1 - beta0), n_infected):
                health_states[i] = 1  # I
    
    # Store final states
    graph.vs["health_state"] = list(health_states) + [-1] * graph["n_bots"]

# ==================== MODIFIED EXISTING FUNCTIONS ====================

def disease_dynamic_step(graph):
    """Main disease step with model selection"""
    disease_model = graph["disease_model"] if "disease_model" in graph.attributes() else "SIR"
    
    if disease_model == "SIRV":
        disease_dynamic_step_SIRV(graph)
    else:  # SIR (default)
        disease_dynamic_step_SIR(graph)

def disease_dynamic_step_SIR(graph):
    """Original SIR disease step (renamed for clarity)"""
    update_susceptibilities_discrete(graph)
    run_SIR_step(graph)

# ==================== EXISTING CODE (unchanged) ====================

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