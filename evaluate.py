import time
import random
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

import config
from environment import GridEnvironment
from astar import a_star_search
from planner import TransformerPlanner
from utils import plot_grid_with_paths

def run_evaluation(args):
    """Runs evaluation comparing A* and Transformer planner."""
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
         # Need patch_size and target_grid_size the model was trained with.
         # Ideally load from checkpoint args, otherwise use config/args.
         # Assuming args match the trained model for now.
         planner = TransformerPlanner(
             model_path=model_load_path,
             patch_size=args.patch_size,
             target_grid_size=args.grid_size,
             device=device
         )
         if planner.model is None: # Check if loading failed internally
             planner = None

    # --- Metrics Storage ---
    metrics = {
        "astar": {"path_lengths": [], "nodes_expanded": [], "times": [], "successes": 0, "failures": 0},
        "transformer": {"path_lengths": [], "steps_taken": [], "times": [], "successes": 0, "failures_stuck": 0, "failures_max_steps":0, "failures_invalid":0, "failures_no_model":0}
    }
    results_data = [] # Store detailed results per case

    # --- Evaluation Loop ---
    print(f"Evaluating on {args.num_cases} test cases (Grid: {args.grid_size}x{args.grid_size}, Obstacles: {args.obstacle_density}, Maze: {args.use_maze})")
    pbar = tqdm(range(args.num_cases))
    for i in pbar:
        # Generate a new test environment
        test_env = GridEnvironment(size=args.grid_size, obstacle_density=args.obstacle_density)
        if args.use_maze:
            test_env.grid = test_env.generate_maze()
            # Ensure final size matches args.grid_size if maze generation adjusted it
            if test_env.size != args.grid_size:
                 # Log or handle this case if needed
                 pass

        start_node, goal_node = test_env.get_random_start_goal_pair(min_dist=args.grid_size * 0.1)

        if start_node is None or goal_node is None:
            # print(f"Skipping test case {i+1}, couldn't find valid start/goal.")
            metrics["astar"]["failures"] += 1 # Count as failure for baseline too
            metrics["transformer"]["failures_invalid"] += 1
            results_data.append({'case': i, 'start': None, 'goal': None, 'status': 'invalid_setup'})
            continue

        case_result = {'case': i, 'start': start_node, 'goal': goal_node, 'grid_shape': test_env.grid.shape}

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
            time_start_transformer = time.time()
            transformer_path, transformer_steps, status = planner.plan_path(
                test_env, start_node, goal_node, max_steps=args.max_steps
            )
            transformer_time = time.time() - time_start_transformer

            case_result['transformer'] = {'time': transformer_time, 'status': status, 'steps': transformer_steps}
            metrics["transformer"]["times"].append(transformer_time)

            if status == "success":
                transformer_len = len(transformer_path) - 1
                metrics["transformer"]["successes"] += 1
                metrics["transformer"]["path_lengths"].append(transformer_len)
                metrics["transformer"]["steps_taken"].append(transformer_steps)
                case_result['transformer'].update({'length': transformer_len})
                # Store paths for plotting successful examples
                if astar_path and len(results_data) < 10 : # Store paths for first few successes
                     case_result['astar_path'] = astar_path
                     case_result['transformer_path'] = transformer_path

            elif status == "failure_stuck":
                metrics["transformer"]["failures_stuck"] += 1
            elif status == "failure_max_steps":
                metrics["transformer"]["failures_max_steps"] += 1
            elif status == "failure_invalid_start_goal": # Should be caught earlier
                metrics["transformer"]["failures_invalid"] += 1
        else: # Planner not loaded
             metrics["transformer"]["failures_no_model"] += 1
             case_result['transformer'] = {'status': 'failure_model_not_loaded', 'time': 0, 'steps': 0, 'length': None}

        results_data.append(case_result)
        pbar.set_postfix({
            'A* OK': metrics["astar"]["successes"],
            'TF OK': metrics["transformer"]["successes"],
            'TF Stuck': metrics["transformer"]["failures_stuck"],
            'TF Steps': metrics["transformer"]["failures_max_steps"]
            })


    # --- Print Aggregate Metrics ---
    print("\n--- Evaluation Results Summary ---")
    num_valid_cases = args.num_cases - metrics["astar"]["failures"] - (metrics["transformer"]["failures_invalid"] if planner else 0) # Cases where start/goal was valid
    num_astar_success = metrics["astar"]["successes"]
    num_transformer_success = metrics["transformer"]["successes"]

    print(f"Total Test Cases Attempted: {args.num_cases}")
    print(f"Valid Start/Goal Pairs Found: {num_valid_cases}")

    print(f"\nA* Success Rate (on valid): {num_astar_success / num_valid_cases * 100 if num_valid_cases > 0 else 0:.2f}% ({num_astar_success}/{num_valid_cases})")
    if num_astar_success > 0:
        avg_astar_len = np.mean(metrics['astar']['path_lengths'])
        avg_astar_nodes = np.mean(metrics['astar']['nodes_expanded'])
        avg_astar_time = np.mean(metrics['astar']['times']) * 1000
        print(f"  Avg A* Path Length: {avg_astar_len:.2f}")
        print(f"  Avg A* Nodes Expanded: {avg_astar_nodes:.2f}")
        print(f"  Avg A* Time: {avg_astar_time:.2f} ms")

    if planner:
        print(f"\nTransformer Success Rate (on valid): {num_transformer_success / num_valid_cases * 100 if num_valid_cases > 0 else 0:.2f}% ({num_transformer_success}/{num_valid_cases})")
        if num_transformer_success > 0:
            avg_transformer_len = np.mean(metrics['transformer']['path_lengths'])
            avg_transformer_steps = np.mean(metrics['transformer']['steps_taken'])
            avg_transformer_time = np.mean(metrics['transformer']['times']) * 1000
            print(f"  Avg Transformer Path Length: {avg_transformer_len:.2f}")
            print(f"  Avg Transformer Steps Taken: {avg_transformer_steps:.2f}")
            print(f"  Avg Transformer Time: {avg_transformer_time:.2f} ms")

            # Calculate path length ratio only on commonly solved instances
            common_astar_lens = []
            common_transformer_lens = []
            for res in results_data:
                if res.get('astar', {}).get('status') == 'success' and \
                   res.get('transformer', {}).get('status') == 'success':
                   common_astar_lens.append(res['astar']['length'])
                   common_transformer_lens.append(res['transformer']['length'])

            if common_astar_lens:
                 path_ratio = np.mean(np.array(common_transformer_lens) / np.array(common_astar_lens))
                 print(f"  Path Length Ratio (Transformer/A* on common successes): {path_ratio:.3f}")
            else:
                 print(f"  Path Length Ratio: N/A (no common successes)")


        print(f"  Transformer Failures (Stuck): {metrics['transformer']['failures_stuck']}")
        print(f"  Transformer Failures (Max Steps): {metrics['transformer']['failures_max_steps']}")
        print(f"  Transformer Failures (Invalid S/G): {metrics['transformer']['failures_invalid']}")
    else:
        print("\nTransformer planner was not loaded/available for evaluation.")


    # --- Save Detailed Results ---
    results_filename = os.path.join(args.results_dir, f"evaluation_results_{args.grid_size}x{args.grid_size}_{'maze' if args.use_maze else 'random'}.pkl")
    with open(results_filename, 'wb') as f:
        pickle.dump({'summary_metrics': metrics, 'detailed_results': results_data}, f)
    print(f"\nDetailed evaluation results saved to {results_filename}")

    # --- Plotting Aggregate Bar Chart ---
    if planner and num_astar_success > 0 and num_transformer_success > 0 :
        labels = ['Avg Path Length', 'Avg Nodes/Steps', 'Avg Time (ms)']
        astar_means = [avg_astar_len, avg_astar_nodes, avg_astar_time]
        transformer_means = [avg_transformer_len, avg_transformer_steps, avg_transformer_time] # Use averages calculated above

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, astar_means, width, label='A* (on success)')
        rects2 = ax.bar(x + width/2, transformer_means, width, label='Transformer (on success)')

        ax.set_ylabel('Scores')
        ax.set_title(f'Comparison (Successful Runs) - {args.grid_size}x{args.grid_size} {"Maze" if args.use_maze else "Random"}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')

        fig.tight_layout()
        plot_filename = os.path.join(args.results_dir, f"comparison_bar_chart_{args.grid_size}x{args.grid_size}_{'maze' if args.use_maze else 'random'}.png")
        plt.savefig(plot_filename)
        print(f"Comparison plot saved to {plot_filename}")
        # plt.show() # Comment out for Slurm

    # --- Plotting Example Paths ---
    # Add code here to load environments and plot paths for a few successful examples from results_data if desired


def parse_eval_args():
    parser = argparse.ArgumentParser(description="Evaluate Pathfinding Models")
    parser.add_argument('--num_cases', type=int, default=config.NUM_TEST_ENVIRONMENTS, help='Number of test environments')
    parser.add_argument('--grid_size', type=int, default=config.TARGET_GRID_SIZE, help='Grid size for testing')
    parser.add_argument('--patch_size', type=int, default=config.PATCH_SIZE, help='Patch size model expects')
    parser.add_argument('--obstacle_density', type=float, default=config.OBSTACLE_DENSITY, help='Obstacle density')
    parser.add_argument('--use_maze', action='store_true', default=config.USE_MAZE_GENERATION, help='Use maze generation for test environments')
    parser.add_argument('--no_maze', action='store_false', dest='use_maze', help='Use random obstacles instead of mazes')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained transformer model .pth file (defaults to best model in config.MODEL_SAVE_DIR)')
    parser.add_argument('--max_steps', type=int, default=config.MAX_PLANNER_STEPS, help='Max steps for transformer planner')
    parser.add_argument('--results_dir', type=str, default=config.EVAL_RESULTS_DIR, help='Directory to save evaluation results')
    return parser.parse_args()

if __name__ == "__main__":
    eval_args = parse_eval_args()
    run_evaluation(eval_args)
