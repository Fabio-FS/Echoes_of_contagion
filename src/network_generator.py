import igraph as ig
import numpy as np
from typing import Optional, Dict, Any
import pickle
import os
from numba import jit

def simulations(param):
    all_results = []
    for i in range(param["n_of_replicas"]):
        print(f"Running replica {i+1}/{param['n_of_replicas']}")
        RES = single_simulations_arrays(param)
        all_results.append(RES)
    return all_results

def single_simulations_arrays(param):
    """Main simulation with arrays instead of Post objects"""
    g = initialize_system(param)
    post_values, post_upvotes, post_readers = initialize_posts_arrays(g)
    RES = initialize_results(g, param)
    
    for step in range(param["N_steps_total"]):
        opinion_dynamic_step_arrays(g, post_values, post_upvotes, post_readers, step)
        if step >= param["waiting_time"]:
            disease_dynamic_step(g)
        update_RES(RES, g, step)

    return RES

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
    


def initialize_posts_arrays(g):
    """Initialize posts as numpy arrays with initial bot upvotes applied"""
    n_users = g["n_humans"]
    N_past = g["post_history"]
    
    # ==================== ARRAY INITIALIZATION ====================
    # Three arrays to replace Post objects
    post_values = np.zeros((n_users, N_past))
    post_upvotes = np.zeros((n_users, N_past), dtype=int)
    # For readers: each post tracks which users have read it
    post_readers = np.zeros((n_users, N_past, len(g.vs)), dtype=bool)
    
    # ==================== FILL WITH USER OPINIONS ====================
    # Initialize all historical posts with user opinions (as if they posted consistently)
    for i in range(n_users):
        post_values[i, :] = g.vs[i]['opinion']
    
    # ==================== APPLY INITIAL BOT UPVOTES ====================
    # Give posts the upvotes they would have received from bots historically
    
    # STEP 1: Check if any bots exist
    if g["n_bots"] > 0:
        bot_threshold = g['bot_threshold']
        
        # STEP 2: Find ALL posts that meet bot upvoting criteria (fully vectorized)
        # Shape: (n_humans, N_past) boolean mask - True where bots would upvote
        should_upvote = post_values < bot_threshold
        
        # STEP 3: Get precomputed bot neighbor counts for all humans
        # Shape: (n_humans,) - number of bot neighbors per human
        bot_neighbor_counts = g["bot_neighbor_counts"]
        
        # STEP 4: Broadcast bot neighbor counts to match post array shape
        # Shape: (n_humans, 1) -> (n_humans, N_past) via broadcasting
        # Each human's bot neighbor count applies to all their historical posts
        bot_upvotes_per_post = bot_neighbor_counts[:, np.newaxis]
        
        # STEP 5: Apply bot upvotes to all qualifying historical posts (vectorized)
        # Element-wise multiplication: where should_upvote is True, add bot_upvotes_per_post
        initial_bot_upvotes = should_upvote * bot_upvotes_per_post
        
        # STEP 6: Update post_upvotes array with initial bot contributions
        post_upvotes += initial_bot_upvotes
    
    return post_values, post_upvotes, post_readers


def opinion_dynamic_step_arrays(graph, post_values, post_upvotes, post_readers, step):
    """Main opinion dynamics step - vectorized array version"""
    generate_posts_arrays(graph, post_values, post_upvotes, post_readers)
    #if step == 40:
    #    selection_user_ids, selection_post_ids = select_posts_vectorized_debug(graph, post_upvotes,post_readers, post_values)
    #else:
    selection_user_ids, selection_post_ids = select_posts_vectorized(graph, post_upvotes,post_readers)
    update_beliefs_and_human_upvotes(graph, post_values, post_upvotes, post_readers, selection_user_ids, selection_post_ids)
    upvotes_bots_only(graph, post_values, post_upvotes)



def generate_posts_arrays(graph, post_values, post_upvotes, post_readers):
    t = graph["current_time_index"]
    
    # Reset the current time slot
    post_upvotes[:, t] = 0
    post_readers[:, t, :] = False
    
    # Generate new posts with noise
    noise = np.random.normal(0, graph["communication_error"], size=graph["n_humans"])
    post_values[:, t] = np.clip(graph.vs['opinion'][:graph["n_humans"]] + noise, -1, 1)
    
    graph["current_time_index"] = (t + 1) % graph["post_history"]



def select_posts_vectorized_debug(graph, post_upvotes, post_readers, post_values):
    """
    Select top k posts for each user based on upvotes and reading history.
    Shows detailed debug info for user 0.
    """
    DEBUG = True
    
    # ==================== SETUP PHASE ====================
    k = graph["feed_size"]           
    n_users = len(graph.vs)          
    n_humans = graph["n_humans"]     
    
    # ==================== GLOBAL POST SORTING ====================
    flat_upvotes = post_upvotes.flatten() + np.random.uniform(0, 0.01, size=post_upvotes.size)
    sort_indices = np.argsort(-flat_upvotes)
    
    n_posts = post_upvotes.shape[1]  
    sorted_users = sort_indices // n_posts  
    sorted_posts = sort_indices % n_posts   
    
    # ==================== MAIN SELECTION LOOP ====================
    selection_user_ids = np.full((n_users, k), -1, dtype=int)
    selection_post_ids = np.full((n_users, k), -1, dtype=int)
    
    for user_i in range(n_users):
        is_from_neighbor = graph["neighbor_lookup"][user_i, sorted_users]
        neighbor_posts_users = sorted_users[is_from_neighbor]  
        neighbor_posts_times = sorted_posts[is_from_neighbor] 
        
        is_read = post_readers[neighbor_posts_users, neighbor_posts_times, user_i]
        unread_mask = ~is_read  
        
        # DEBUG: Show detailed info for user 0
        if DEBUG and user_i == 0:
            print(f"\n=== USER 0 DEBUG ===")
            
            # 1) Show ALL unread posts accessible to user 0
            unread_users_all = neighbor_posts_users[unread_mask]
            unread_times_all = neighbor_posts_times[unread_mask]
            
            print(f"\nALL UNREAD POSTS ACCESSIBLE TO USER 0 ({len(unread_users_all)} total):")
            print("PostUser | PostTime | Upvotes | Opinion")
            print("-" * 40)
            for i in range(len(unread_users_all)):
                user_id = unread_users_all[i]
                time_id = unread_times_all[i]
                upvotes = post_upvotes[user_id, time_id]
                opinion = post_values[user_id, time_id]
                print(f"{user_id:8d} | {time_id:8d} | {upvotes:7.0f} | {opinion:7.3f}")

            print("number of neighbors: ", len(graph.vs[0]["human_neighbors"]))
        
        # Select top k unread posts
        unread_users = neighbor_posts_users[unread_mask][:k]  
        unread_posts = neighbor_posts_times[unread_mask][:k]  
        
        n_selected = len(unread_users)  
        selection_user_ids[user_i, :n_selected] = unread_users
        selection_post_ids[user_i, :n_selected] = unread_posts
        
        # DEBUG: Show what user 0 actually reads
        if DEBUG and user_i == 0:
            print(f"\nPOSTS USER 0 ACTUALLY READS THIS ROUND ({n_selected} posts):")
            print("PostUser | PostTime | Upvotes | Opinion")
            print("-" * 40)
            for j in range(n_selected):
                user_id = unread_users[j]
                time_id = unread_posts[j]
                upvotes = post_upvotes[user_id, time_id]
                opinion = post_values[user_id, time_id]
                print(f"{user_id:8d} | {time_id:8d} | {upvotes:7.0f} | {opinion:7.3f}")
    
    return selection_user_ids, selection_post_ids


def select_posts_vectorized(graph, post_upvotes, post_readers):
    """
    Select top k posts for each user based on upvotes and reading history.
    
    This function implements a feed algorithm where:
    1. All posts are sorted globally by upvotes (most popular first)
    2. Each user gets a personalized feed of k unread posts from their neighbors
    3. Only posts from human neighbors are considered (bots don't create posts)
    
    Args:
        graph: igraph object containing network structure and user data
        post_upvotes: numpy array (n_humans, post_history) - upvote counts for each post
        post_readers: numpy array (n_humans, post_history, n_total_users) - tracks who read what
    
    Returns:
        selection_user_ids: array (n_total_users, k) - which users created the selected posts
        selection_post_ids: array (n_total_users, k) - which time indices of those posts
        (-1 values indicate no more posts available for that user)
    """
    DEBUG = False
    # ==================== SETUP PHASE ====================
    # Extract key parameters from graph
    k = graph["feed_size"]           # Number of posts per user's feed (e.g., 2)
    n_users = len(graph.vs)          # Total users: humans + bots (e.g., 60)
    n_humans = graph["n_humans"]     # Only humans create posts (e.g., 50)
    
    # ==================== GLOBAL POST SORTING ====================
    # Instead of each user sorting posts separately (expensive), 
    # we sort ALL posts once and reuse this ordering
    
    # Flatten the 2D post_upvotes array into 1D for sorting
    # Shape: (n_humans, post_history) -> (n_humans * post_history,)
    flat_upvotes = post_upvotes.flatten() + np.random.uniform(0, 0.01, size=post_upvotes.size)
    
    # Sort by upvotes (descending order, hence the negative sign)
    # This gives us indices into the flattened array
    sort_indices = np.argsort(-flat_upvotes)
    
    # Convert flat indices back to (user_id, time_index) coordinates
    n_posts = post_upvotes.shape[1]  # Number of time steps in post history
    sorted_users = sort_indices // n_posts  # Which user created each post
    sorted_posts = sort_indices % n_posts   # Which time step each post was created
    
    # Now sorted_users[0] and sorted_posts[0] give us the most upvoted post
    # sorted_users[1] and sorted_posts[1] give us the second most upvoted, etc.
    
    # ==================== NEIGHBOR DATA EXTRACTION ====================
    # Get pre-computed neighbor lists to avoid expensive igraph access in loops
    # This was computed once in precompute_neighbors() and stored in the graph
    all_neighbors = graph["all_human_neighbors_list"]
    
    # ==================== RESULT ARRAYS INITIALIZATION ====================
    # Initialize output arrays with -1 (indicates "no post available")
    # Shape: (n_total_users, k) - each user gets k post slots
    selection_user_ids = np.full((n_users, k), -1, dtype=int)
    selection_post_ids = np.full((n_users, k), -1, dtype=int)
    
    # ==================== NEIGHBOR LOOKUP TABLE ====================
    # Build a fast lookup table: neighbor_lookup[user_i, neighbor_j] = True
    # This avoids using np.isin() which is slow for repeated queries
    # Shape: (n_total_users, n_humans) - each row shows who that user follows
    #neighbor_lookup = np.zeros((n_users, n_humans), dtype=bool)
    
    # Populate the lookup table
    #for i in range(n_users):
    #    if len(all_neighbors[i]) > 0:  # Skip users with no neighbors (e.g., isolated nodes)
    #        # Set True for each neighbor of user i
    #        neighbor_lookup[i, all_neighbors[i]] = True
    
    # ==================== MAIN SELECTION LOOP ====================
    # For each user, find their top k unread posts from neighbors
    for user_i in range(n_users):
        
        # STEP 1: Filter globally sorted posts to only this user's neighbors
        # Use the lookup table for fast neighbor checking
        # is_from_neighbor[j] = True if sorted_users[j] is a neighbor of user_i
        #is_from_neighbor = neighbor_lookup[user_i, sorted_users]
        is_from_neighbor = graph["neighbor_lookup"][user_i, sorted_users]#is_from_neighbor = np.isin(sorted_users, list(graph["neighbor_sets"][user_i]))

        
        # Extract the subset of posts that are from this user's neighbors
        # These maintain the global sorting order (most upvoted first)
        neighbor_posts_users = sorted_users[is_from_neighbor]  # Who created these posts
        neighbor_posts_times = sorted_posts[is_from_neighbor]  # When they were created
        
        # STEP 2: Filter out posts this user has already read
        # Check reading history: has user_i read each of these neighbor posts?
        # post_readers[post_user, post_time, reader] = True if reader read that post
        is_read = post_readers[neighbor_posts_users, neighbor_posts_times, user_i]
        unread_mask = ~is_read  # Flip to get unread posts
        
        # STEP 3: Take the top k unread posts (they're already sorted by upvotes)
        unread_users = neighbor_posts_users[unread_mask][:k]  # Who created the top k
        unread_posts = neighbor_posts_times[unread_mask][:k]  # When they were created
        
        # STEP 4: Store results in output arrays
        n_selected = len(unread_users)  # Might be less than k if not enough posts
        selection_user_ids[user_i, :n_selected] = unread_users
        selection_post_ids[user_i, :n_selected] = unread_posts
        # Remaining slots stay as -1 (no more posts available)
    


    
    if DEBUG:
        # ==================== DEBUG SECTION ====================
        # Check for problems: users who didn't get enough posts
        has_negative = np.any(selection_user_ids < 0)
        if has_negative:
            print(f"Found -1 values in selection!")
            print(f"n_users: {n_users}, k: {k}")
            print(f"post_upvotes shape: {post_upvotes.shape}")
            
            # Identify which users have missing posts
            users_with_neg = np.any(selection_user_ids < 0, axis=1)
            problem_users = np.where(users_with_neg)[0]
            print(f"Users with -1s: {problem_users[:10]}...")  # Show first 10
            
            # Examine the first problematic user
            if len(problem_users) > 0:
                user_i = problem_users[0]
                neighbors = all_neighbors[user_i]
                print(f"User {user_i} has {len(neighbors)} neighbors: {neighbors}")
                
                # Check if their neighbors have posts available
                if len(neighbors) > 0:
                    neighbor_posts = post_upvotes[neighbors, :]
                    print(f"Neighbor posts shape: {neighbor_posts.shape}")
                    print(f"Total neighbor posts available: {neighbor_posts.size}")
    
    # ==================== RETURN RESULTS ====================
    return selection_user_ids, selection_post_ids


def update_beliefs_and_human_upvotes(graph, post_values, post_upvotes, post_readers, selection_user_ids, selection_post_ids):
    """Sequential reading with belief updates AND human upvoting combined"""
    n_humans = graph["n_humans"]
    mu = graph['mu']
    epsilon = graph['epsilon']
    k = selection_user_ids.shape[1]  # Number of posts per user (e.g., 5)
    
    # Get all current opinions at once (vectorized)
    current_opinions = np.array(graph.vs['opinion'][:n_humans])
    
    # OUTER LOOP: Process each post position sequentially
    for post_slot in range(k):
        # STEP 1: Get data for this slot - assume all users have posts
        post_user_indices = selection_user_ids[:n_humans, post_slot]
        post_time_indices = selection_post_ids[:n_humans, post_slot]
        
        # STEP 2: Handle the rare case of missing posts
        valid_posts = post_user_indices >= 0

            
        # STEP 3: Get post opinions (vectorized, with masking for invalid posts)
        post_opinions = np.where(valid_posts, 
                                post_values[post_user_indices, post_time_indices], 
                                current_opinions)  # Use current opinion as dummy for invalid posts
        
        # STEP 4 & 5: Use JIT-compiled BCM calculations
        opinion_diffs = post_opinions - current_opinions
        within_confidence = np.abs(opinion_diffs) < epsilon
        
        # Apply belief updates (vectorized)
        opinion_updates = mu * opinion_diffs * within_confidence * valid_posts
    
        #print(f"Updates applied: {np.sum(opinion_updates != 0)}")
        #print(f"Opinion range: {np.min(current_opinions):.3f} to {np.max(current_opinions):.3f}")
        # Return both the updates and the upvote mask
        human_upvotes = within_confidence * valid_posts  # Boolean mask for which posts get upvoted
        
        # STEP 6: Apply human upvotes to the posts (vectorized)
        if np.any(human_upvotes):
            # Get coordinates of posts that humans upvote
            upvoting_users = post_user_indices[human_upvotes]
            upvoting_times = post_time_indices[human_upvotes]
            
            # Convert to flat indices for bincount
            flat_indices = upvoting_users * post_upvotes.shape[1] + upvoting_times
            upvote_counts = np.bincount(flat_indices, minlength=post_upvotes.size)
            upvote_counts = upvote_counts.reshape(post_upvotes.shape)
            post_upvotes += upvote_counts
        
        # STEP 7: Apply belief updates
        current_opinions += opinion_updates
        
        # STEP 8: Mark posts as read (vectorized)
        if np.any(valid_posts):
            valid_indices = np.where(valid_posts)[0]
            post_readers[post_user_indices[valid_posts], post_time_indices[valid_posts], valid_indices] = True
    
    # STEP 9: Update graph with final opinions (including bot padding)
    graph.vs["opinion"] = list(current_opinions) + [0] * graph["n_bots"]



def upvotes_bots_only(graph, post_values, post_upvotes):
    """Fully vectorized bot upvoting - no loops at all"""
    
    # ==================== PARAMETER EXTRACTION ====================
    n_bots = graph["n_bots"]                    # Total bots in system - needed to check if any exist
    bot_threshold = graph['bot_threshold']      # Opinion threshold below which bots upvote (-0.5 typically)
    n_humans = graph["n_humans"]               # Number of humans (for array slicing)
    
    # ==================== TIME INDEX CALCULATION ====================
    # Find the time slot where posts were just created this round
    current_time = graph["current_time_index"] - 1  
    if current_time < 0:
        current_time = graph["post_history"] - 1
    
    # ==================== EARLY EXIT OPTIMIZATION ====================
    if n_bots == 0:
        return
    
    # ==================== FULLY VECTORIZED PROCESSING ====================
    
    # STEP 1: Get ALL latest post opinions at once (vectorized array slice)
    # Shape: (n_humans,) - one opinion value per human
    # Advantage: Single array access instead of loop with n_humans accesses
    latest_opinions = post_values[:n_humans, current_time]
    
    # STEP 2: Find ALL posts that meet bot criteria (vectorized comparison)
    # Shape: (n_humans,) boolean mask - True where bots should upvote
    # Advantage: NumPy's C-level comparison, much faster than Python loop
    should_upvote = latest_opinions < bot_threshold
    
    # STEP 3: Use precomputed bot neighbor counts (NO list comprehension!)
    # Shape: (n_humans,) - number of bot neighbors per human
    # Advantage: Pure array lookup, precomputed during initialization
    bot_neighbor_counts = graph["bot_neighbor_counts"]
    
    # STEP 4: Calculate upvotes for ALL qualifying posts (vectorized multiplication)
    # Shape: (n_humans,) - upvotes to add per human's post
    # Advantage: Element-wise operations instead of conditional logic in loop
    upvotes_to_add = should_upvote * bot_neighbor_counts
    
    # STEP 5: Apply ALL upvotes at once (vectorized addition)
    # Advantage: Single array operation instead of multiple individual updates
    # Only updates posts that actually get upvotes (where upvotes_to_add > 0)
    post_upvotes[:n_humans, current_time] += upvotes_to_add



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


def initialize_results(g, param):
    RES = {
        'opinions': np.zeros((param["N_steps"], g["n_humans"])),
        'health_states': np.zeros((param["N_steps"], g["n_humans"])),
        'mean_opinion': np.zeros(param["N_steps"]),
        'opinion_var': np.zeros(param["N_steps"]),
        'fraction_ever_infected': np.zeros(param["N_steps"])  # Changed this line
    }
    return RES

def update_RES(RES, g, step):
    # Save key data
    RES['opinions'][step] = g.vs['opinion'][:g["n_humans"]]
    RES['health_states'][step] = g.vs['health_state'][:g["n_humans"]]
    RES['mean_opinion'][step] = np.mean(g.vs['opinion'][:g["n_humans"]])
    RES['opinion_var'][step] = np.var(g.vs['opinion'][:g["n_humans"]])
    
    # Track cumulative infections: I(t) + R(t) = N - S(t)
    health_states = np.array(g.vs['health_state'][:g["n_humans"]])
    n_susceptible = np.sum(health_states == 0)
    RES['fraction_ever_infected'][step] = (g["n_humans"] - n_susceptible) / g["n_humans"]  # Changed this line
    


def save_results(all_results, param, filename=None):
    """Save results from multiple replicas and parameters to file"""
    if filename is None:
        filename = f"sim_n{param['n_humans']}_bots{param['n_bots']}_reps{param['n_of_replicas']}.pkl"
    
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    
    data = {
        'all_results': all_results,  # List of replica results
        'parameters': param
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Results from {len(all_results)} replicas saved to {filepath}")

def load_results(filename):
    """Load results from file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['all_results'], data['parameters']
    """Load results from file"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['all_results'], data['parameters']