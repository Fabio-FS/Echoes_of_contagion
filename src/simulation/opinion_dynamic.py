import numpy as np
import igraph as ig
from .posts import generate_posts_arrays
from .feed_algorithms import select_posts_vectorized


def opinion_dynamic_step_arrays(graph, post_values, post_upvotes, post_readers, step):
    """Main opinion dynamics step - vectorized array version"""
    generate_posts_arrays(graph, post_values, post_upvotes, post_readers)
    #if step == 40:
    #    selection_user_ids, selection_post_ids = select_posts_vectorized_debug(graph, post_upvotes,post_readers, post_values)
    #else:
    selection_user_ids, selection_post_ids = select_posts_vectorized(graph, post_upvotes,post_readers)
    update_beliefs_and_human_upvotes(graph, post_values, post_upvotes, post_readers, selection_user_ids, selection_post_ids)
    upvotes_bots_only(graph, post_values, post_upvotes)


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


