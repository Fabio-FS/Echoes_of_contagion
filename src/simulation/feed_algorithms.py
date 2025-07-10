import numpy as np
import igraph as ig

def select_posts_vectorized(graph, post_upvotes, post_readers):
    """Main dispatcher function"""
    feed_algo = graph["feed_algorithm"] if "feed_algorithm" in graph.attributes() else "popularity"
    
    if feed_algo == "popularity":
        return select_posts_popularity(graph, post_upvotes, post_readers)
    elif feed_algo == "random":
        return select_posts_random(graph, post_upvotes, post_readers)
    elif feed_algo == "similarity":
        return select_posts_similarity(graph, post_upvotes, post_readers)
    else:
        raise ValueError(f"Unknown feed algorithm: {feed_algo}")



def select_posts_popularity(graph, post_upvotes, post_readers):
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



def select_posts_random(graph, post_upvotes, post_readers):
    """
    Vectorized random post selection with per-user randomization.
    """
    # ==================== SETUP PHASE ====================
    k = graph["feed_size"]           
    n_users = len(graph.vs)          
    n_humans = graph["n_humans"]
    n_posts = post_upvotes.shape[1]
    
    # ==================== RESULT ARRAYS INITIALIZATION ====================
    selection_user_ids = np.full((n_users, k), -1, dtype=int)
    selection_post_ids = np.full((n_users, k), -1, dtype=int)
    
    # ==================== VECTORIZED PROCESSING ====================
    
    # Pre-create all possible posts
    all_users = np.repeat(np.arange(n_humans), n_posts)
    all_times = np.tile(np.arange(n_posts), n_humans)
    
    for user_i in range(n_users):
        
        # STEP 1: Filter to neighbor posts (vectorized)
        is_from_neighbor = graph["neighbor_lookup"][user_i, all_users]
        neighbor_users = all_users[is_from_neighbor]
        neighbor_times = all_times[is_from_neighbor]
        
        if len(neighbor_users) == 0:
            continue
        
        # STEP 2: Filter out read posts (vectorized)
        is_read = post_readers[neighbor_users, neighbor_times, user_i]
        unread_mask = ~is_read
        unread_users = neighbor_users[unread_mask]
        unread_times = neighbor_times[unread_mask]
        
        # STEP 3: Random shuffle THIS USER'S available posts
        if len(unread_users) > 0:
            perm = np.random.permutation(len(unread_users))
            n_selected = min(k, len(unread_users))
            
            selection_user_ids[user_i, :n_selected] = unread_users[perm[:n_selected]]
            selection_post_ids[user_i, :n_selected] = unread_times[perm[:n_selected]]
    
    return selection_user_ids, selection_post_ids

def select_posts_similarity(graph, post_upvotes, post_readers):
    pass

