import time
import random
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import glob # For finding files
from pathlib import Path # For expanding ~

import config # Default config
from environment import GridEnvironment
from astar import a_star_search
from planner import TransformerPlanner
from utils import plot_grid_with_paths, is_valid as util_is_valid # For find_random_free_cell_on_map

# --- Map File Parser ---
def parse_map_file(filepath):
    """
    Parses a .map file from the MovingAI benchmark format.
    Returns a 2D numpy array (0 for traversable, 1 for obstacle) and its dimensions.
    Obstacles: @, O, T (Trees, Water are often obstacles too)
    Traversable: ., G, S, W (Swamp might have higher cost in some games, here treated as traversable)
    """
    lines = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    height, width = 0, 0
    map_data_started = False
    grid_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("height"):
            height = int(line.split(" ")[1])
        elif line.startswith("width"):
            width = int(line.split(" ")[1])
        elif line == "map":
            map_data_started = True
        elif map_data_started and height > 0 and width > 0:
            if len(grid_lines) < height:
                grid_lines.append(line)

    if not grid_lines or len(grid_lines) != height or (height > 0 and len(grid_lines[0]) != width):
        # print(f"Warning: Map file {filepath} format error or incomplete. Expected {height}x{width}, got {len(grid_lines)} lines, first line len {len(grid_lines[0]) if grid_lines else 0}")
        return None, 0, 0

    grid = np.zeros((height, width), dtype=np.int8)
    for r in range(height):
        for c in range(width):
            char = grid_lines[r][c]
            if char in ['@', 'O', 'T']: # Obstacles
                grid[r, c] = 1
            elif char in ['.', 'G', 'S', 'W']: # Traversable
                grid[r, c] = 0
            else:
                # print(f"Warning: Unknown character '{char}' in map {filepath} at ({r},{c}). Treating as obstacle.")
                grid[r, c] = 1 # Treat unknown as obstacle
    return grid, height, width

def find_random_free_cell_on_map(grid_array):
    """Finds a random non-obstacle cell from a given grid_array."""
    free_cells = list(zip(*np.where(grid_array == 0)))
    if not free_cells:
        return None
    return random.choice(free_cells)

def get_random_start_goal_for_map(grid_array, min_dist_factor=0.1):
    """Finds a random pair of non-obstacle start and goal cells on a map."""
    height, width = grid_array.shape
    min_dist = max(1, int(min(height, width) * min_dist_factor))

    start = find_random_free_cell_on_map(grid_array)
    goal = find_random_free_cell_on_map(grid_array)
    attempts = 0
    while goal is None or start is None or start == goal or np.linalg.norm(np.array(start) - np.array(goal)) < min_dist:
        if attempts > 100:
            # print(f"Warning: Could not find suitable start/goal pair for map after 100 attempts.")
            return None, None
        start = find_random_free_cell_on_map(grid_array)
        goal = find_random_free_cell_on_map(grid_array)
        attempts += 1
    return start, goal


# --- Evaluation Function (Modified) ---
def run_evaluation(args):
    print(f"\n--- Running Evaluation ---")
    print(f"Args: {args}")

    device = config.DEVICE
    os.makedirs(args.results_dir, exist_ok=True)

    # --- Load Planner ---
    model_load_path = args.model_path or os.path.join(config.MODEL_SAVE_DIR, os.path.basename(config.BEST_MODEL_FILE))
    if not os.path.exists(model_load_path):
         print(f"ERROR: Model file not found at {model_load_path}. Cannot evaluate planner.")
         planner = None
    else:
         # Use args.model_patch_size and args.model_coord_size for the planner
         planner = TransformerPlanner(
             model_path=model_load_path,
             patch_size=args.model_patch_size,       # Patch size the MODEL was trained with
             target_grid_size=args.model_coord_size, # Coordinate size the MODEL was trained with
             device=device
         )
         if planner.model is None:
             planner = None

    # --- Metrics Storage ---
    metrics = {
        "astar": {"path_lengths": [], "nodes_expanded": [], "times": [], "successes": 0, "failures": 0},
        "transformer": {"path_lengths": [], "steps_taken": [], "times": [], "successes": 0,
                        "failures_stuck": 0, "failures_max_steps":0, "failures_invalid":0,
                        "failures_no_model":0, "failures_coord_clipping": 0}
    }
    results_data = []
    num_maps_processed = 0
    num_scenarios_attempted_on_maps = 0

    # --- Determine Test Environments ---
    test_environments_info = [] # List of (env_name, grid_data_or_None_for_random_gen)

    if args.use_game_grids:
        print(f"Loading game grids from: {args.game_grids_dir}")
        game_map_dir_expanded = Path(args.game_grids_dir).expanduser()
        map_files = glob.glob(str(game_map_dir_expanded / '**/*.map'), recursive=True)
        print(f"Found {len(map_files)} .map files.")
        if not map_files:
            print("No .map files found. Exiting.")
            return

        for map_file_path in map_files:
            map_grid, map_h, map_w = parse_map_file(map_file_path)
            if map_grid is not None:
                # For each map, run multiple scenarios (start/goal pairs)
                for i in range(args.num_scenarios_per_map):
                    test_environments_info.append({
                        "name": Path(map_file_path).name + f"_scen{i+1}",
                        "grid_data": map_grid,
                        "grid_h": map_h,
                        "grid_w": map_w,
                        "source": "game_map"
                    })
            else:
                print(f"Skipping invalid map file: {map_file_path}")
        num_maps_processed = len(map_files)
        print(f"Prepared {len(test_environments_info)} scenarios from game maps.")
    else:
        print(f"Generating {args.num_cases} random/maze environments (Grid: {args.eval_grid_size}x{args.eval_grid_size})")
        for i in range(args.num_cases):
            test_environments_info.append({
                "name": f"random_env_{i+1}",
                "grid_data": None, # Will be generated
                "grid_h": args.eval_grid_size,
                "grid_w": args.eval_grid_size,
                "source": "generated"
            })

    if not test_environments_info:
        print("No environments to evaluate. Exiting.")
        return

    # --- Evaluation Loop ---
    pbar = tqdm(test_environments_info, desc="Evaluating Environments")
    for env_info in pbar:
        pbar.set_description(f"Evaluating: {env_info['name'][:30]}")

        if env_info["source"] == "game_map":
            test_env = GridEnvironment(size=env_info["grid_h"], grid_data=env_info["grid_data"]) # Size is from map
            start_node, goal_node = get_random_start_goal_for_map(test_env.grid)
            num_scenarios_attempted_on_maps +=1
        else: # source == "generated"
            test_env = GridEnvironment(size=args.eval_grid_size, obstacle_density=args.obstacle_density)
            if args.use_maze_generation:
                test_env.grid = test_env.generate_maze()
            start_node, goal_node = test_env.get_random_start_goal_pair(min_dist=args.eval_grid_size * 0.1)

        if start_node is None or goal_node is None:
            metrics["astar"]["failures"] += 1
            metrics["transformer"]["failures_invalid"] += 1
            results_data.append({'case_name': env_info['name'], 'start': None, 'goal': None, 'status': 'invalid_setup'})
            continue

        case_result = {'case_name': env_info['name'], 'start': start_node, 'goal': goal_node, 'grid_shape': test_env.grid.shape}

        # Check if start/goal are outside the model's trained coordinate range if planner is active
        coord_warning_issued = False
        if planner:
            if start_node[0] >= args.model_coord_size or start_node[1] >= args.model_coord_size or \
               goal_node[0] >= args.model_coord_size or goal_node[1] >= args.model_coord_size:
                # This warning is for information; the planner's clamping handles it.
                # print(f"Warning: Case {env_info['name']} S/G {start_node}/{goal_node} exceeds model's coord_size {args.model_coord_size}. Coords will be clamped.")
                metrics["transformer"]["failures_coord_clipping"] += 1 # Count how often this happens
                coord_warning_issued = True


        # --- A* (Baseline) ---
        time_start_astar = time.time()
        astar_path, astar_nodes = a_star_search(test_env, start_node, goal_node)
        astar_time = time.time() - time_start_astar
        case_result['astar'] = {'time': astar_time}
        if astar_path:
            astar_len = len(astar_path) - 1
            metrics["astar"]["successes"] += 1
            metrics["astar"]["path_lengths"].append(astar_len)
            metrics["astar"]["nodes_expanded"].append(astar_nodes)
            metrics["astar"]["times"].append(astar_time)
            case_result['astar'].update({'status': 'success', 'length': astar_len, 'nodes': astar_nodes})
        else:
            metrics["astar"]["failures"] += 1
            case_result['astar'].update({'status': 'failure', 'length': None, 'nodes': astar_nodes})

        # --- Transformer Planner ---
        if planner:
            # Override max_steps for planner if it's a large game map
            current_max_steps = args.max_steps
            if env_info["source"] == "game_map":
                 current_max_steps = max(args.max_steps, env_info["grid_h"] * env_info["grid_w"] // 2) # Heuristic

            time_start_transformer = time.time()
            transformer_path, transformer_steps, status = planner.plan_path(
                test_env, start_node, goal_node, max_steps=current_max_steps
            )
            transformer_time = time.time() - time_start_transformer

            case_result['transformer'] = {'time': transformer_time, 'status': status, 'steps': transformer_steps, 'coord_clipping_occurred': coord_warning_issued}
            metrics["transformer"]["times"].append(transformer_time)

            if status == "success":
                transformer_len = len(transformer_path) - 1
                metrics["transformer"]["successes"] += 1
                metrics["transformer"]["path_lengths"].append(transformer_len)
                metrics["transformer"]["steps_taken"].append(transformer_steps)
                case_result['transformer'].update({'length': transformer_len})
            elif status == "failure_stuck":
                metrics["transformer"]["failures_stuck"] += 1
            elif status == "failure_max_steps":
                metrics["transformer"]["failures_max_steps"] += 1
            elif status == "failure_invalid_start_goal":
                metrics["transformer"]["failures_invalid"] += 1
        else:
             metrics["transformer"]["failures_no_model"] += 1
             case_result['transformer'] = {'status': 'failure_model_not_loaded', 'time': 0, 'steps': 0, 'length': None}

        results_data.append(case_result)
        # Update progress bar less frequently if many scenarios
        if len(pbar) % 10 == 0 or len(pbar) < 20:
            pbar.set_postfix({
                'A* OK': metrics["astar"]["successes"],
                'TF OK': metrics["transformer"]["successes"],
            })

    # --- Print Aggregate Metrics ---
    print("\n--- Evaluation Results Summary ---")
    total_evaluated_scenarios = len(test_environments_info)
    num_astar_success = metrics["astar"]["successes"]
    num_transformer_success = metrics["transformer"]["successes"]

    print(f"Total Scenarios Evaluated: {total_evaluated_scenarios}")
    if args.use_game_grids:
        print(f"  From {num_maps_processed} unique map files, {num_scenarios_attempted_on_maps} S/G pairs attempted.")
        if planner:
            print(f"  Transformer scenarios with coordinate clipping: {metrics['transformer']['failures_coord_clipping']}")


    print(f"\nA* Success Rate: {num_astar_success / total_evaluated_scenarios * 100 if total_evaluated_scenarios > 0 else 0:.2f}% ({num_astar_success}/{total_evaluated_scenarios})")
    avg_astar_len, avg_astar_nodes, avg_astar_time = 0,0,0
    if num_astar_success > 0:
        avg_astar_len = np.mean(metrics['astar']['path_lengths'])
        avg_astar_nodes = np.mean(metrics['astar']['nodes_expanded'])
        avg_astar_time = np.mean(metrics['astar']['times']) * 1000
        print(f"  Avg A* Path Length (on success): {avg_astar_len:.2f}")
        print(f"  Avg A* Nodes Expanded (on success): {avg_astar_nodes:.2f}")
        print(f"  Avg A* Time (on success): {avg_astar_time:.2f} ms")

    avg_transformer_len, avg_transformer_steps, avg_transformer_time = 0,0,0
    if planner:
        print(f"\nTransformer Success Rate: {num_transformer_success / total_evaluated_scenarios * 100 if total_evaluated_scenarios > 0 else 0:.2f}% ({num_transformer_success}/{total_evaluated_scenarios})")
        if num_transformer_success > 0:
            avg_transformer_len = np.mean(metrics['transformer']['path_lengths'])
            avg_transformer_steps = np.mean(metrics['transformer']['steps_taken'])
            avg_transformer_time = np.mean(metrics['transformer']['times']) * 1000
            print(f"  Avg Transformer Path Length (on success): {avg_transformer_len:.2f}")
            print(f"  Avg Transformer Steps Taken (on success): {avg_transformer_steps:.2f}")
            print(f"  Avg Transformer Time (on success): {avg_transformer_time:.2f} ms")

            common_astar_lens = []
            common_transformer_lens = []
            for res in results_data:
                if res.get('astar', {}).get('status') == 'success' and \
                   res.get('transformer', {}).get('status') == 'success':
                   common_astar_lens.append(res['astar']['length'])
                   common_transformer_lens.append(res['transformer']['length'])
            if common_astar_lens:
                 path_ratio = np.mean(np.array(common_transformer_lens) / (np.array(common_astar_lens) + 1e-6)) # Add epsilon for safety
                 print(f"  Path Length Ratio (TF/A* on common successes): {path_ratio:.3f}")

        print(f"  Transformer Failures (Stuck): {metrics['transformer']['failures_stuck']}")
        print(f"  Transformer Failures (Max Steps): {metrics['transformer']['failures_max_steps']}")
        print(f"  Transformer Failures (Invalid S/G): {metrics['transformer']['failures_invalid']}")
    else:
        print("\nTransformer planner was not loaded/available for evaluation.")

    # --- Save Detailed Results ---
    eval_type = "game_grids" if args.use_game_grids else \
                ("maze" if args.use_maze_generation else "random")
    results_filename = os.path.join(args.results_dir, f"eval_results_{eval_type}.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump({'summary_metrics': metrics, 'detailed_results': results_data, 'args': vars(args)}, f)
    print(f"\nDetailed evaluation results saved to {results_filename}")

    # --- Plotting Aggregate Bar Chart ---
    if planner and num_astar_success > 0 and num_transformer_success > 0 :
        labels = ['Avg Path Length', 'Avg Nodes/Steps', 'Avg Time (ms)']
        astar_means = [avg_astar_len, avg_astar_nodes, avg_astar_time]
        transformer_means = [avg_transformer_len, avg_transformer_steps, avg_transformer_time]

        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, astar_means, width, label='A* (on success)')
        rects2 = ax.bar(x + width/2, transformer_means, width, label='Transformer (on success)')
        ax.set_ylabel('Scores')
        ax.set_title(f'Comparison (Successful Runs) - {eval_type.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')
        fig.tight_layout()
        plot_filename = os.path.join(args.results_dir, f"comparison_chart_{eval_type}.png")
        plt.savefig(plot_filename)
        print(f"Comparison plot saved to {plot_filename}")

def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate Pathfinding Models")
    # General evaluation settings
    parser.add_argument('--results_dir', type=str, default=config.EVAL_RESULTS_DIR, help='Directory to save evaluation results')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained transformer .pth file (defaults to best model in config.MODEL_SAVE_DIR)')
    parser.add_argument('--max_steps', type=int, default=config.MAX_PLANNER_STEPS, help='Default max steps for transformer planner (can be overridden for large maps)')

    # Arguments for MODEL that was TRAINED
    parser.add_argument('--model_patch_size', type=int, default=config.PATCH_SIZE, help='Patch size the LOADED MODEL was trained with.')
    parser.add_argument('--model_coord_size', type=int, default=config.TARGET_GRID_SIZE, help='Max coordinate size (grid dimension) the LOADED MODEL was trained for.')

    # Group for selecting evaluation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--use-random-grids', action='store_true', help='Evaluate on newly generated random obstacle grids.')
    mode_group.add_argument('--use-maze-grids', action='store_true', help='Evaluate on newly generated maze grids.')
    mode_group.add_argument('--use-game-grids', action='store_true', help='Evaluate on .map files from game benchmarks.')

    # Settings for --use-random-grids or --use-maze-grids
    parser.add_argument('--num_cases', type=int, default=config.NUM_TEST_ENVIRONMENTS, help='Number of random/maze test environments if not using --use-game-grids.')
    parser.add_argument('--eval_grid_size', type=int, default=config.TARGET_GRID_SIZE, help='Grid size for generated random/maze test environments.')
    parser.add_argument('--obstacle_density', type=float, default=config.OBSTACLE_DENSITY, help='Obstacle density for random grids.')
    # Note: use_maze_generation is implicitly handled by --use-maze-grids

    # Settings for --use-game-grids
    parser.add_argument('--game_grids_dir', type=str, default="~/Datasets/a_star_maps/", help='Directory containing .map game grid files.')
    parser.add_argument('--num_scenarios_per_map', type=int, default=5, help='Number of random start/goal scenarios to run per game map.')

    args = parser.parse_args()

    # Consolidate flags for simpler logic later
    if args.use_random_grids:
        args.eval_mode = "random"
        args.use_maze_generation = False # Explicitly false
    elif args.use_maze_grids:
        args.eval_mode = "maze"
        args.use_maze_generation = True # Explicitly true
    elif args.use_game_grids:
        args.eval_mode = "game"
        args.use_maze_generation = False # Not applicable
    else:
        # Should not happen due to mutually_exclusive_group
        raise ValueError("An evaluation mode must be selected.")

    return args

if __name__ == "__main__":
    eval_args = parse_eval_args()
    run_evaluation(eval_args)
