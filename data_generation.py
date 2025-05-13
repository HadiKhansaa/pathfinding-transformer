import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import glob
from pathlib import Path
import math

import config
from environment import GridEnvironment
from astar import a_star_search
from utils import delta_to_action
from evaluate import parse_map_file # Re-use the parser from evaluate.py

def get_map_files(directory):
    """Finds all .map files recursively in a directory."""
    expanded_dir = Path(directory).expanduser()
    if not expanded_dir.is_dir():
        print(f"Error: Game map directory not found: {expanded_dir}")
        return []
    map_files = glob.glob(str(expanded_dir / '**/*.map'), recursive=True)
    print(f"Found {len(map_files)} game map files in {expanded_dir}.")
    return map_files

def generate_trajectory_on_env(grid_env, num_trajectories_per_env=1, min_dist_factor=0.1): # Parameter name is correct here
    """
    Generates a specified number of A* trajectories for a given environment instance.
    Returns a list of state-action pairs from all successful trajectories.
    """
    trajectories_from_this_env = []
    successful_paths = 0
    for _ in range(num_trajectories_per_env):
        # The min_dist calculation is now inside get_random_start_goal_pair
        # We just need to pass the factor.

        # Corrected call:
        start_node, goal_node = grid_env.get_random_start_goal_pair(min_dist_factor=min_dist_factor)

        if start_node is None or goal_node is None:
            # print(f"Could not find valid S/G for current env. Skipping one trajectory.")
            continue

        path, _ = a_star_search(grid_env, start_node, goal_node)

        if path and len(path) > 1:
            successful_paths += 1
            grid_repr = grid_env.grid # Get grid array once
            for i in range(len(path) - 1):
                current_pos = path[i]
                next_pos = path[i+1]
                state = (grid_repr, current_pos, goal_node) # Patch extracted by Dataset
                dr = next_pos[0] - current_pos[0]
                dc = next_pos[1] - current_pos[1]
                action_idx = delta_to_action(dr, dc)
                trajectories_from_this_env.append((state, action_idx))
    return trajectories_from_this_env

def generate_and_save_data_combined(
    dataset_type="train", # "train" or "val"
    # For randomly generated maps:
    num_random_envs=0,
    target_grid_size_random=100, # Size for random/maze generated grids
    obstacle_density_random=0.3,
    use_maze_random=False,
    # For game maps:
    game_map_filepaths=None, # List of specific game map files to use
    trajectories_per_game_map=1,
    # Common:
    output_filename_prefix="dataset"
    ):
    """
    Generates trajectories from a mix of sources (randomly generated and/or game maps)
    and saves them.
    """
    all_state_action_pairs = []
    max_dim_encountered = 0 # To track max coordinate for COORD_VOCAB_SIZE recommendation

    desc = f"Generating {dataset_type} data"
    total_envs_to_process = num_random_envs + (len(game_map_filepaths) if game_map_filepaths else 0)
    pbar = tqdm(total=total_envs_to_process, desc=desc)

    # 1. Process Game Maps (if any)
    if game_map_filepaths and trajectories_per_game_map > 0:
        print(f"Processing {len(game_map_filepaths)} game maps for {dataset_type} set...")
        for map_file in game_map_filepaths:
            pbar.set_postfix_str(f"Map: {Path(map_file).name[:20]}")
            map_grid_array, map_h, map_w = parse_map_file(map_file)
            if map_grid_array is not None:
                max_dim_encountered = max(max_dim_encountered, map_h, map_w)
                env_game_map = GridEnvironment(grid_data=map_grid_array, height=map_h, width=map_w)
                trajs = generate_trajectory_on_env(env_game_map, trajectories_per_game_map)
                all_state_action_pairs.extend(trajs)
            else:
                print(f"Skipping invalid game map: {map_file}")
            pbar.update(1)

    # 2. Process Randomly Generated Maps (if any)
    if num_random_envs > 0:
        print(f"Generating {num_random_envs} random/maze maps ({target_grid_size_random}x{target_grid_size_random}) for {dataset_type} set...")
        max_dim_encountered = max(max_dim_encountered, target_grid_size_random)
        for i in range(num_random_envs):
            pbar.set_postfix_str(f"Random Env: {i+1}/{num_random_envs}")
            # Create env for random maps (assuming square for simplicity here)
            env_random = GridEnvironment(height=target_grid_size_random,
                                         width=target_grid_size_random,
                                         obstacle_density=obstacle_density_random)
            if use_maze_random:
                env_random.generate_maze() # generate_maze uses env's height/width

            # For random maps, let's generate 1 trajectory per environment instance by default
            trajs = generate_trajectory_on_env(env_random, num_trajectories_per_env=1)
            all_state_action_pairs.extend(trajs)
            pbar.update(1)
    pbar.close()

    print(f"\nGenerated a total of {len(all_state_action_pairs)} state-action pairs for {dataset_type} set.")
    if max_dim_encountered > 0:
        print(f"IMPORTANT: Max grid dimension encountered: {max_dim_encountered-1}. "
              f"Ensure config.MODEL_COORD_VOCAB_SIZE is at least {max_dim_encountered}.")


    # Save the data
    final_filename = f"{config.EXPERT_DATA_DIR}/{output_filename_prefix}_{dataset_type}.pkl"
    os.makedirs(os.path.dirname(final_filename), exist_ok=True)
    with open(final_filename, 'wb') as f:
        pickle.dump(all_state_action_pairs, f)
    print(f"Saved {dataset_type} data to {final_filename}")
    return all_state_action_pairs, final_filename


# (load_data function remains the same)
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
    os.makedirs(config.EXPERT_DATA_DIR, exist_ok=True)

    game_maps_all = []
    if config.USE_GAME_MAPS_FOR_TRAINING:
        game_maps_all = get_map_files(config.GAME_MAPS_DATA_DIR)
        random.shuffle(game_maps_all) # Shuffle for random split

    # Split game maps for training and validation
    num_game_maps_total = len(game_maps_all)
    num_game_maps_train = math.ceil(num_game_maps_total * config.GAME_MAPS_TRAIN_RATIO) # Use ceil for train
    game_maps_train_set = game_maps_all[:num_game_maps_train]
    game_maps_val_set = game_maps_all[num_game_maps_train:]

    print(f"Total game maps: {num_game_maps_total}")
    print(f"Using {len(game_maps_train_set)} game maps for training set.")
    print(f"Using {len(game_maps_val_set)} game maps for validation set.")

    # --- Generate Training Data ---
    print("\n--- Generating Training Data ---")
    train_game_map_files_to_use = game_maps_train_set if config.USE_GAME_MAPS_FOR_TRAINING else None
    num_random_train_to_gen = config.NUM_RANDOM_TRAIN_ENVS if config.USE_RANDOM_MAPS_FOR_TRAINING else 0

    if not train_game_map_files_to_use and num_random_train_to_gen == 0:
        print("No data sources configured for training. Skipping training data generation.")
    else:
        generate_and_save_data_combined(
            dataset_type="train",
            num_random_envs=num_random_train_to_gen,
            target_grid_size_random=config.TARGET_GRID_SIZE, # Size for randomly generated grids
            obstacle_density_random=config.OBSTACLE_DENSITY,
            use_maze_random=config.USE_MAZE_GENERATION, # For random portion
            game_map_filepaths=train_game_map_files_to_use,
            trajectories_per_game_map=config.TRAJECTORIES_PER_GAME_MAP,
            output_filename_prefix="combined_train_data"
        )

    # --- Generate Validation Data ---
    print("\n--- Generating Validation Data ---")
    val_game_map_files_to_use = game_maps_val_set if config.USE_GAME_MAPS_FOR_TRAINING else None
    num_random_val_to_gen = config.NUM_RANDOM_VAL_ENVS if config.USE_RANDOM_MAPS_FOR_TRAINING else 0

    if not val_game_map_files_to_use and num_random_val_to_gen == 0:
        print("No data sources configured for validation. Skipping validation data generation.")
    else:
        generate_and_save_data_combined(
            dataset_type="val",
            num_random_envs=num_random_val_to_gen,
            target_grid_size_random=config.TARGET_GRID_SIZE,
            obstacle_density_random=config.OBSTACLE_DENSITY,
            use_maze_random=config.USE_MAZE_GENERATION,
            game_map_filepaths=val_game_map_files_to_use,
            trajectories_per_game_map=config.TRAJECTORIES_PER_GAME_MAP, # Can be fewer for val
            output_filename_prefix="combined_val_data"
        )

    print("\nData generation process complete.")