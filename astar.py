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

    Args:
        grid_env (GridEnvironment): The environment object.
        start (tuple): Start coordinates (r, c).
        goal (tuple): Goal coordinates (r, c).

    Returns:
        tuple: (path, expanded_nodes_count)
               path is a list of (r, c) tuples, or None if not found.
               expanded_nodes_count is the number of nodes popped from the priority queue.
    """
    if grid_env.is_obstacle(start[0], start[1]) or grid_env.is_obstacle(goal[0], goal[1]):
        return None, 0 # Start or goal is obstacle

    pq = [(0 + octile_distance(start, goal), 0, start, [start])]  # (f_score, g_score, node, path_list)
    visited = {start: 0} # node: g_score
    # For faster lookups, store parent pointers if path reconstruction is done separately
    # came_from = {start: None}
    expanded_nodes_count = 0

    while pq:
        _, g, current_node, path = heapq.heappop(pq)
        expanded_nodes_count += 1

        if current_node == goal:
            return path, expanded_nodes_count

        # Check if we found a shorter path to this node already (due to heap property)
        # This can happen if a node is pushed multiple times with different g-scores.
        if g > visited[current_node]:
            continue

        for neighbor_node, cost_to_reach, _ in grid_env.get_neighbors(current_node[0], current_node[1]):
            new_g = g + cost_to_reach
            if neighbor_node not in visited or new_g < visited[neighbor_node]:
                visited[neighbor_node] = new_g
                h = octile_distance(neighbor_node, goal)
                f_new = new_g + h
                new_path = path + [neighbor_node] # Simple path extension
                heapq.heappush(pq, (f_new, new_g, neighbor_node, new_path))
                # came_from[neighbor_node] = current_node # If reconstructing path later

    return None, expanded_nodes_count # Path not found
