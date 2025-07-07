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


    # opinion bits
    g.vs["opinion"]  = np.random.uniform(-1, 1, len(g.vs))      # for speed I give an opinion also to bots. it won't be used.
    
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
    
    g["beta0"] = param["beta0"] #0.1
    g["recovery_rate"] = param["recovery_rate"]


    g["waiting_time"] = param["waiting_time"]


    # select I0 initial infected
    indx_inf = np.random.choice(param["n_humans"], param["I0"], replace=False)

    for i in indx_inf:
        g.vs[i]["health_state"] = 1
    g.vs["is_bot"] = [False] * param["n_humans"] + [True] * param["n_bots"]

