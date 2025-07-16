import numpy as np
import igraph as ig



def initialize_system(param):
    g = create_network(param)
    add_bots(g, param)
    precompute_neighbors(g, param)
    initialize_users_and_bots(g, param)
    #g["meshgrids"] = precompute_meshgrids(g, param["post_history"])
    g["current_time_index"] = 0

    
    


    
    neighbor_lookup = np.zeros((len(g.vs), g["n_humans"]), dtype=bool)
    for i in range(len(g.vs)):
        if len(g.vs[i]["human_neighbors"]) > 0:
            neighbor_lookup[i, g.vs[i]["human_neighbors"]] = True
    g["neighbor_lookup"] = neighbor_lookup
    #g["neighbor_sets"] = [set(g.vs[i]["human_neighbors"]) for i in range(len(g.vs))]

    return g







def create_network(param):
    
    """Create Watts-Strogatz network with initial attributes."""
    n=param["n_humans"]
    nei=param["nei"]
    p=param["p"]

    g = ig.Graph.Watts_Strogatz(dim = 1, size = n, nei = nei, p = p, loops=False, multiple=False)
    # https://python.igraph.org/en/stable/api/igraph.GraphBase.html#Watts_Strogatz

    return g
def add_bots(g, param):
    """Add bots as new nodes with connections to existing network."""
    
    g.add_vertices(param["n_bots"])
    
    # Set node types
    user_type = ["human"] * param["n_humans"] + ["bot"] * param["n_bots"]
    g.vs["user_type"] = user_type
    
    avg_degree = int(np.mean(g.degree()))
    
    # Vectorized edge creation
    bot_ids = np.arange(param["n_humans"], param["n_humans"] + param["n_bots"])
    bot_ids = np.repeat(bot_ids, avg_degree)  # Each bot ID repeated avg_degree times
    
    # Generate targets for each bot separately, then concatenate
    all_targets = []
    for _ in range(param["n_bots"]):
        targets = np.random.choice(param["n_humans"], avg_degree, replace=False)
        all_targets.extend(targets)
    
    # Create all edges at once
    edges_to_add = list(zip(bot_ids, all_targets))
    g.add_edges(edges_to_add)
def precompute_neighbors(g, param):
    """Store neighbor lists as node attributes for each node AND precompute bot neighbor counts."""
    n_humans = param["n_humans"]
    
    # Initialize empty lists for all nodes
    g.vs["human_neighbors"] = [np.array([], dtype=int) for _ in range(g.vcount())]
    g.vs["bot_neighbors"] = [np.array([], dtype=int) for _ in range(g.vcount())]
    g.vs["all_neighbors"] = [np.array([], dtype=int) for _ in range(g.vcount())]

        
    for node_id in range(g.vcount()):
        neighbors = np.array(g.neighbors(node_id))
        
        human_neighs = neighbors[neighbors < n_humans]
        bot_neighs = neighbors[neighbors >= n_humans]
        
        g.vs[node_id]["human_neighbors"] = human_neighs
        g.vs[node_id]["bot_neighbors"] = bot_neighs
        g.vs[node_id]["all_neighbors"] = neighbors
    
    g["all_human_neighbors_list"] = [g.vs[i]["human_neighbors"] for i in range(g.vcount())]
    
    # ==================== PRECOMPUTE BOT NEIGHBOR COUNTS ====================
    # Store bot neighbor counts for all humans as numpy array for vectorized access
    # Shape: (n_humans,) - eliminates need for list comprehension in upvoting
    g["bot_neighbor_counts"] = np.array([len(g.vs[i]["bot_neighbors"]) for i in range(n_humans)])


def initialize_users_and_bots(g, param):
    # general bits
    g["n_humans"] = param["n_humans"]
    g["n_bots"] = param["n_bots"]

    # opinion bits - NEW FLEXIBLE INITIALIZATION
    g.vs["opinion"] = initialize_opinions(param, len(g.vs))
    
    g["communication_error"] = param["communication_error"]
    g["mu"] = param["mu"] # 0.1
    g["epsilon"] = param["epsilon"] #0.3
    g["bot_threshold"] = param["bot_threshold"]
    g["post_history"] = param["post_history"]
    g["feed_size"] = param["feed_size"]

    # opinion to behavior bits
    g["O0"] = 0.0 
    g["behavior_strength"] = 2.0





    # disease bits
    g.vs['health_state'] = [0] * param["n_humans"] + [-1] * param["n_bots"]  # All human start susceptible, all bots start bots. # S=0, I=1, R=2, Bot=-1


    g["disease_model"] = param["disease_model"] if "disease_model" in param.attributes() else "SIR"
    if(g["disease_model"] == "SIRV"):
        g["xi_max"] = param["xi_max"]
        g["use_discrete_vaccination"] = param["use_discrete_vaccination"]
        g["vaccination_groups"] = param["vaccination_groups"]

    g["beta0"] = param["beta0"] #0.1
    g["recovery_rate"] = param["recovery_rate"]
    g["waiting_time"] = param["waiting_time"]





    # select I0 initial infected
    indx_inf = np.random.choice(param["n_humans"], param["I0"], replace=False)
    for i in indx_inf:
        g.vs[i]["health_state"] = 1
    g.vs["is_bot"] = [False] * param["n_humans"] + [True] * param["n_bots"]


def initialize_opinions(param, n_total):
    """Initialize opinions based on param settings"""
    n_humans = param["n_humans"]
    
    # Check what type of initialization to use
    if "opinion_init_type" not in param:
        # Default: uniform distribution
        opinions = np.random.uniform(-1, 1, n_total)
    
    elif param["opinion_init_type"] == "gaussian":
        # Single Gaussian N(M, S)
        M = param.get("opinion_mean", 0.0)
        S = param.get("opinion_std", 0.5)
        opinions = np.random.normal(M, S, n_total)
        opinions = np.clip(opinions, -1, 1)  # Keep in valid range
    
    elif param["opinion_init_type"] == "bimodal":
        # Sum of two Gaussians
        M1 = param.get("opinion_mean1", -0.5)
        S1 = param.get("opinion_std1", 0.2)
        M2 = param.get("opinion_mean2", 0.5)
        S2 = param.get("opinion_std2", 0.2)
        weight1 = param.get("opinion_weight1", 0.5)  # Fraction from first Gaussian
        
        n_from_first = int(n_humans * weight1)
        n_from_second = n_humans - n_from_first
        
        opinions_humans = np.concatenate([
            np.random.normal(M1, S1, n_from_first),
            np.random.normal(M2, S2, n_from_second)
        ])
        np.random.shuffle(opinions_humans)  # Mix the two groups
        opinions_humans = np.clip(opinions_humans, -1, 1)
        
        # Add bot opinions (can be random or fixed)
        bot_opinions = np.random.uniform(-1, 1, n_total - n_humans)
        opinions = np.concatenate([opinions_humans, bot_opinions])
    
    else:
        # Fallback to uniform
        opinions = np.random.uniform(-1, 1, n_total)
    
    return opinions