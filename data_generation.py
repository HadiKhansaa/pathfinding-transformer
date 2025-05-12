import os
import pickle
import random
from tqdm import tqdm
import numpy as np

import config
from environment import GridEnvironment
from astar import a_star_search
from utils import delta_to_action

def generate_expert_trajectory(grid_env):
    """Generates a single A* trajectory for a given environment."""
    start_node, goal_node = grid_env.get_random_start_goal_pair(min_dist=config.TARGET_GRID_SIZE * 0.1) # Ensure some distance

    if start_node is None or goal_node is None:
        return None # Failed to find valid start/goal

    path, _ = a_star_search(grid_env, start_node, goal_node)

    if path and len(path) > 1:
        trajectory = []
        grid_repr = grid_env.grid # Get grid array once
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i+1]

            # State: (grid_array, current_pos, goal_pos)
            # We extract the patch later in the Dataset __getitem__
            state = (grid_repr, current_pos, goal_node)

            dr = next_pos[0] - current_pos[0]
            dc = next_pos[1] - current_pos[1]
            action_idx = delta_to_action(dr, dc)
            trajectory.append((state, action_idx))
        return trajectory
    else:
        return None # Path not found or trivial

def generate_and_save_data(num_trajectories, filename, grid_size, obstacle_density, use_maze):
    """Generates multiple trajectories and saves them to a file."""
    all_state_action_pairs = []
    pbar = tqdm(range(num_trajectories), desc=f"Generating data for {filename}")
    successful_envs = 0

    while successful_envs < num_trajectories and len(pbar) > 0: # Loop until enough successful envs or pbar finishes
        env = GridEnvironment(size=grid_size, obstacle_density=obstacle_density)
        if use_maze:
            # Ensure maze generation results in the target grid size if possible
            env.grid = env.generate_maze()
            if env.size != grid_size:
                 # Handle potential size mismatch from maze generation if needed
                 # print(f"Warning: Maze generation changed grid size to {env.size}. Adjusting...")
                 # This might involve cropping or padding, ensure it makes sense for your use case.
                 # For simplicity, we assume generate_maze is adjusted or we accept its output size
                 # Or better: generate slightly larger and crop/pad in generate_maze itself.
                 # Let's assume env.generate_maze handles this to return grid_size x grid_size
                 pass


        trajectory = generate_expert_trajectory(env)
        if trajectory:
            all_state_action_pairs.extend(trajectory)
            successful_envs += 1
            pbar.update(1)
            pbar.set_postfix({"States": len(all_state_action_pairs)})
        else:
             # Environment might be unsolvable or start/goal invalid
             # Don't count this towards num_trajectories, effectively increases attempts
             # pbar doesn't advance here automatically
             pass


    print(f"\nGenerated {len(all_state_action_pairs)} state-action pairs from {successful_envs} successful A* runs.")

    # Save the data
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(all_state_action_pairs, f)
    print(f"Saved data to {filename}")

def load_data(filename):
    """Loads trajectory data from a file."""
    if not os.path.exists(filename):
        print(f"Error: Data file not found at {filename}")
        return None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} state-action pairs from {filename}")
    return data

if __name__ == "__main__":
    print("Generating Training Data...")
    generate_and_save_data(
        num_trajectories=config.NUM_TRAIN_TRAJECTORIES,
        filename=config.TRAIN_DATA_FILE,
        grid_size=config.TARGET_GRID_SIZE,
        obstacle_density=config.OBSTACLE_DENSITY,
        use_maze=config.USE_MAZE_GENERATION
    )

    print("\nGenerating Validation Data...")
    generate_and_save_data(
        num_trajectories=config.NUM_VAL_TRAJECTORIES,
        filename=config.VAL_DATA_FILE,
        grid_size=config.TARGET_GRID_SIZE,
        obstacle_density=config.OBSTACLE_DENSITY,
        use_maze=config.USE_MAZE_GENERATION
    )

    print("\nData generation complete.")
