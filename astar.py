import numpy as np
import heapq
import config # For ACTION_COSTS

def octile_distance(p1, p2):
    """Calculates the Octile distance heuristic."""
    dr = abs(p1[0] - p2[0])
    dc = abs(p1[1] - p2[1])
    return (dr + dc) + (np.sqrt(2) - 2) * min(dr, dc)



def a_star_search(grid_env, start, goal):
    """
    Performs A* search on the given grid environment.

    Returns:
        tuple: (path, expanded_nodes_count, expanded_nodes_set)
               path is a list of (r, c) tuples, or None if not found.
               expanded_nodes_count is the number of unique nodes popped from the priority queue.
               expanded_nodes_set is a set of (r,c) tuples for all expanded nodes.
    """
    # print(f"  [A* DEBUG] Entering a_star_search. Start: {start} (type: {type(start[0])}), Goal: {goal} (type: {type(goal[0])})")
    # print(f"  [A* DEBUG] Grid dimensions: H={grid_env.height}, W={grid_env.width}")

    if not grid_env.is_valid(start[0], start[1]):
        # print(f"  [A* DEBUG] Start node {start} is invalid.")
        return None, 0, set()
    if not grid_env.is_valid(goal[0], goal[1]):
        # print(f"  [A* DEBUG] Goal node {goal} is invalid.")
        return None, 0, set()

    if grid_env.is_obstacle(start[0], start[1]):
        # print(f"  [A* DEBUG] Start node {start} is an obstacle.")
        return None, 0, set()
    if grid_env.is_obstacle(goal[0], goal[1]):
        # print(f"  [A* DEBUG] Goal node {goal} is an obstacle.")
        return None, 0, set()

    # print(f"  [A* DEBUG] Start and Goal are valid and not obstacles.")

    pq = []  # Priority queue: (f_score, g_score, tie_breaker_id, current_node, path_list)
    tie_breaker_id = 0 # To ensure unique items in PQ if f_scores are same

    # g_scores: cost from start to a node
    g_scores = {start: 0}
    # f_scores: estimated total cost from start to goal through a node (g_score + heuristic)
    # We can store f_scores in the PQ directly. No need for a separate f_scores dict if g_scores is primary.

    # Add start node to PQ
    h_initial = octile_distance(start, goal)
    f_initial = 0 + h_initial
    heapq.heappush(pq, (f_initial, 0, tie_breaker_id, start, [start]))
    tie_breaker_id += 1

    expanded_nodes_set = set() # To store unique nodes popped (expanded)
    # print(f"  [A* DEBUG] Initialized A* loop variables. PQ: {pq}")

    loop_count = 0
    max_loops = grid_env.height * grid_env.width * 2 # Safety break

    while pq:
        loop_count += 1
        if loop_count > max_loops:
            # print(f"  [A* DEBUG] A* loop seems too long ({loop_count} iterations), breaking.")
            break

        try:
            f_score_val, g_val, _, current_node, path = heapq.heappop(pq)
        except Exception as e:
            # print(f"  [A* DEBUG] Error during heapq.heappop: {e}")
            # import traceback; traceback.print_exc()
            return None, len(expanded_nodes_set), expanded_nodes_set

        # Check if we've already processed this node (if it was added to PQ multiple times)
        # This check is implicitly handled by only adding to PQ if a shorter path is found.
        # However, adding to expanded_nodes_set ensures we only count unique expansions.
        if current_node in expanded_nodes_set:
            continue
        expanded_nodes_set.add(current_node)

        if current_node == goal:
            # print(f"  [A* DEBUG] Goal reached: {current_node}")
            return path, len(expanded_nodes_set), expanded_nodes_set

        # print(f"  [A* DEBUG] Expanding node: {current_node} with g_score={g_val}")
        try:
            neighbors_list = grid_env.get_neighbors(current_node[0], current_node[1])
        except Exception as e:
            # print(f"  [A* DEBUG] Error in grid_env.get_neighbors for node {current_node}: {e}")
            # import traceback; traceback.print_exc()
            return None, len(expanded_nodes_set), expanded_nodes_set

        for neighbor_node, cost_to_reach, _ in neighbors_list:
            tentative_g_score = g_val + cost_to_reach

            if tentative_g_score < g_scores.get(neighbor_node, float('inf')):
                # This path to neighbor is better than any previous one. Record it.
                g_scores[neighbor_node] = tentative_g_score
                h_val = octile_distance(neighbor_node, goal)
                f_score_new = tentative_g_score + h_val
                new_path = path + [neighbor_node]
                try:
                    heapq.heappush(pq, (f_score_new, tentative_g_score, tie_breaker_id, neighbor_node, new_path))
                    tie_breaker_id += 1
                except Exception as e:
                    # print(f"  [A* DEBUG] Error during heapq.heappush for neighbor {neighbor_node}: {e}")
                    # import traceback; traceback.print_exc()
                    return None, len(expanded_nodes_set), expanded_nodes_set

    # print(f"  [A* DEBUG] PQ empty or loop terminated, path not found. Expanded nodes: {len(expanded_nodes_set)}")
    return None, len(expanded_nodes_set), expanded_nodes_set