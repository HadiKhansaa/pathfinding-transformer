# run_demo_slurm.py
import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Slurm
import matplotlib.pyplot as plt
import random
import os
from pathlib import Path
import argparse # For command-line arguments

# Import functions/classes from local project files
import config
from environment import GridEnvironment
from astar import a_star_search
from planner import TransformerPlanner
from utils import plot_grid_with_paths_and_explored
from evaluate import parse_map_file

def main(args):
    print(f"Demo using device: {config.DEVICE}")
    print(f"Loading map from: {args.map_file}")
    map_grid_array, map_h, map_w = parse_map_file(args.map_file)

    if map_grid_array is None:
        print(f"Failed to parse map file: {args.map_file}")
        return
    
    print(f"Map loaded: {map_h}x{map_w}")
    demo_env = GridEnvironment(grid_data=map_grid_array, height=map_h, width=map_w)

    # --- Select Start and Goal ---
    start_node, goal_node = None, None
    if args.start_row is not None and args.start_col is not None and \
       args.goal_row is not None and args.goal_col is not None:
        start_node = (args.start_row, args.start_col)
        goal_node = (args.goal_row, args.goal_col)
        # Basic validation
        if not demo_env.is_valid(start_node[0], start_node[1]) or demo_env.is_obstacle(start_node[0], start_node[1]) or \
           not demo_env.is_valid(goal_node[0], goal_node[1]) or demo_env.is_obstacle(goal_node[0], goal_node[1]):
            print(f"Warning: Manual start {start_node} or goal {goal_node} is invalid on the map. Attempting random.")
            start_node, goal_node = demo_env.get_random_start_goal_pair(min_dist_factor=args.min_dist_factor)
    else:
        print(f"Attempting to find random start/goal (min_dist_factor={args.min_dist_factor})...")
        start_node, goal_node = demo_env.get_random_start_goal_pair(min_dist_factor=args.min_dist_factor)

    if start_node is None or goal_node is None:
        print("Could not find valid start/goal for the demo on this map.")
        return
    
    print(f"Using Start: {start_node}, Goal: {goal_node}")

    # --- Run A* ---
    print("\nRunning A*...")
    astar_time_start = time.time()
    astar_path, astar_nodes_expanded_count, astar_expanded_nodes_set = a_star_search(demo_env, start_node, goal_node)
    astar_time_end = time.time()
    print(f"A* Finished in {astar_time_end - astar_time_start:.4f} seconds.")
    if astar_path:
        print(f"  A* Path found. Length: {len(astar_path)-1}, Nodes Expanded: {astar_nodes_expanded_count}")
    else:
        print(f"  A* Path NOT found. Nodes Expanded: {astar_nodes_expanded_count}")

    # --- Load and Run Transformer Planner ---
    print(f"\nLoading Transformer planner with model: {args.model_path}...")
    tf_planner = TransformerPlanner(model_path=args.model_path, device=config.DEVICE)

    transformer_path = None
    transformer_steps_taken = 0
    tf_status = "not_run"
    transformer_visited_set = set()

    if tf_planner.model is None:
        print("Transformer model could not be loaded. Skipping Transformer planning.")
    else:
        print(f"  Planner loaded model with patch_size: {tf_planner.loaded_model_patch_size}, coord_vocab_size: {tf_planner.loaded_model_coord_size}")
        print("Running Transformer Planner...")
        tf_time_start = time.time()
        transformer_path, transformer_steps_taken, tf_status, transformer_visited_set = tf_planner.plan_path(
            demo_env, start_node, goal_node, max_steps=args.max_tf_steps
        )
        tf_time_end = time.time()
        print(f"Transformer Planner Finished in {tf_time_end - tf_time_start:.4f} seconds.")
        print(f"  Transformer Status: {tf_status}")
        if tf_status == "success":
            print(f"  Transformer Path found. Length: {len(transformer_path)-1}, Steps: {transformer_steps_taken} (Nodes Visited: {len(transformer_visited_set)})")
        else:
            print(f"  Transformer Path NOT found or failed. Steps: {transformer_steps_taken} (Nodes Visited: {len(transformer_visited_set)})")

    # --- Plotting ---
    plot_filename = args.output_plot_file
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True) # Ensure output dir exists

    title = f"Map: {Path(args.map_file).name}, S:{start_node}, G:{goal_node}\n"
    if astar_path: title += f"A* Len: {len(astar_path)-1}, Exp: {astar_nodes_expanded_count} | "
    else: title += f"A* Failed, Exp: {astar_nodes_expanded_count} | "

    if tf_planner.model is not None: # Only add TF info if planner was attempted
        if transformer_path and tf_status == "success": title += f"TF Len: {len(transformer_path)-1}, Steps: {transformer_steps_taken} ({len(transformer_visited_set)} visited)"
        else: title += f"TF Status: {tf_status}, Steps: {transformer_steps_taken} ({len(transformer_visited_set)} visited)"
    else:
        title += "TF Not Loaded"

    plot_grid_with_paths_and_explored(
        grid_env=demo_env,
        start=start_node,
        goal=goal_node,
        astar_path=astar_path,
        transformer_path=transformer_path,
        astar_expanded_nodes_set=astar_expanded_nodes_set,
        transformer_visited_set=transformer_visited_set,
        title=title
    )
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close() # Close the figure to free memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A* and Transformer Planner Demo on a map file.")
    parser.add_argument('--map_file', type=str, required=True, help='Path to the .map file (e.g., demo_grid.map)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file (e.g., model_best.pth)')
    parser.add_argument('--output_plot_file', type=str, default="demo_comparison_plot.png", help='Filename for the output plot image.')
    parser.add_argument('--start_row', type=int, default=None, help='Manual start row (optional)')
    parser.add_argument('--start_col', type=int, default=None, help='Manual start col (optional)')
    parser.add_argument('--goal_row', type=int, default=None, help='Manual goal row (optional)')
    parser.add_argument('--goal_col', type=int, default=None, help='Manual goal col (optional)')
    parser.add_argument('--min_dist_factor', type=float, default=0.3, help='Factor for minimum S/G distance if random.')
    parser.add_argument('--max_tf_steps', type=int, default=5000, help='Max steps for Transformer planner.')
    
    cli_args = parser.parse_args()
    main(cli_args)
