import numpy as np
import igraph as ig





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
def generate_posts_arrays(graph, post_values, post_upvotes, post_readers):
    t = graph["current_time_index"]
    
    # Reset the current time slot
    post_upvotes[:, t] = 0
    post_readers[:, t, :] = False
    
    # Generate new posts with noise
    noise = np.random.normal(0, graph["communication_error"], size=graph["n_humans"])
    post_values[:, t] = np.clip(graph.vs['opinion'][:graph["n_humans"]] + noise, -1, 1)
    
    graph["current_time_index"] = (t + 1) % graph["post_history"]