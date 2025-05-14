import numpy as np
import matplotlib.pyplot as plt
import torch
import config

# --- Action Utilities ---
def action_to_delta(action_index):
    """Converts action index to (dr, dc) delta."""
    if 0 <= action_index < len(config.ACTIONS):
        return config.ACTIONS[action_index]
    raise ValueError(f"Invalid action index: {action_index}")

def delta_to_action(dr, dc):
    """Converts (dr, dc) delta to action index."""
    try:
        # Normalize dr/dc in case of floating point inaccuracies if needed
        dr = round(dr)
        dc = round(dc)
        return config.ACTIONS.index((dr, dc))
    except ValueError:
        # Find closest action (simple Euclidean distance for deltas)
        min_dist = float('inf')
        best_action = 0
        target_delta = np.array([dr, dc])
        for i, (adr, adc) in enumerate(config.ACTIONS):
            action_delta = np.array([adr, adc])
            dist = np.linalg.norm(target_delta - action_delta)
            if dist < min_dist:
                min_dist = dist
                best_action = i
        # print(f"Warning: Could not find exact action for delta ({dr}, {dc}). Using closest: {config.ACTIONS[best_action]}")
        return best_action

# --- Grid & Patch Utilities ---
def is_valid(r, c, grid_size):
    """Checks if coordinates are within grid bounds."""
    return 0 <= r < grid_size and 0 <= c < grid_size

def extract_local_patch(grid, center_r, center_c, patch_size):
    """Extracts a square patch centered at (center_r, center_c).

    Handles boundary conditions by padding.

    Args:
        grid (np.ndarray): The 2D grid array.
        center_r (int): Center row coordinate.
        center_c (int): Center column coordinate.
        patch_size (int): The side length of the patch (must be odd).

    Returns:
        np.ndarray: The extracted (and possibly padded) patch of shape (patch_size, patch_size).
    """
    grid_size = grid.shape[0]
    half_patch = patch_size // 2

    # Calculate patch boundaries
    r_start, r_end = center_r - half_patch, center_r + half_patch + 1
    c_start, c_end = center_c - half_patch, center_c + half_patch + 1

    # Create padded grid view if necessary
    pad_top = max(0, -r_start)
    pad_bottom = max(0, r_end - grid_size)
    pad_left = max(0, -c_start)
    pad_right = max(0, c_end - grid_size)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        # Pad the original grid
        padded_grid = np.pad(
            grid,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=config.PATCH_PADDING_VALUE
        )
        # Adjust coordinates for the padded grid
        p_center_r = center_r + pad_top
        p_center_c = center_c + pad_left
        # Extract from padded grid
        patch = padded_grid[p_center_r - half_patch : p_center_r + half_patch + 1,
                            p_center_c - half_patch : p_center_c + half_patch + 1]
    else:
        # Extract directly if fully within bounds
        patch = grid[r_start:r_end, c_start:c_end]

    # Ensure the patch has the correct shape (should always be true with padding)
    if patch.shape != (patch_size, patch_size):
        # This indicates an error in logic, but handle defensively
        # print(f"Warning: Patch extraction resulted in wrong shape {patch.shape}. Expected ({patch_size}, {patch_size}). Padding manually.")
        correct_patch = np.full((patch_size, patch_size), config.PATCH_PADDING_VALUE, dtype=grid.dtype)
        # Attempt to place the extracted patch into the correct shape
        r_offset = (patch_size - patch.shape[0]) // 2
        c_offset = (patch_size - patch.shape[1]) // 2
        try: # Use try-except for robustness if indices are off
            correct_patch[r_offset:r_offset+patch.shape[0], c_offset:c_offset+patch.shape[1]] = patch
        except IndexError:
             pass # Keep the fully padded patch
        return correct_patch

    return patch


# --- Plotting Utilities ---
def plot_grid_with_paths(grid_env, start, goal, astar_path=None, transformer_path=None, title="Grid"):
    """Plots the grid, start, goal, and optionally paths."""
    grid_size = grid_env.size
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'salmon', 'lightgreen', 'blue', 'cyan', 'magenta'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5] # 0:free, 1:obstacle, 2:goal, 3:start
                                                     # 4: A* path, 5: Transformer path, 6: Overlap

    plot_grid = np.copy(grid_env.grid).astype(float)

    # Mark paths first
    if transformer_path:
        for r, c in transformer_path:
            if (r, c) != start and (r, c) != goal:
                plot_grid[r, c] = 5 # Cyan for Transformer path
    if astar_path:
        for r, c in astar_path:
            if (r, c) != start and (r, c) != goal:
                if plot_grid[r, c] == 5: # Check for overlap
                    plot_grid[r, c] = 6 # Magenta for Overlap
                else:
                    plot_grid[r, c] = 4 # Blue for A* path

    # Mark start and goal last to ensure they are visible
    if start: plot_grid[start] = 3 # lightgreen
    if goal:  plot_grid[goal] = 2 # salmon

    plt.figure(figsize=(8, 8))
    plt.imshow(plot_grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=6)

    # Add grid lines
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    tick_step = max(1, grid_size // 10)
    ax.set_xticks(np.arange(0, grid_size, tick_step))
    ax.set_yticks(np.arange(0, grid_size, tick_step))

    plt.title(title, fontsize=10)
    plt.show()




def plot_grid_with_paths_and_explored(grid_env, start, goal, astar_path=None, transformer_path=None,
                                      astar_expanded_nodes_set=None, transformer_visited_set=None,
                                      title="Grid Comparison"):
    grid_h, grid_w = grid_env.height, grid_env.width
    cmap_list = ['white', 'black', 'salmon', 'lightgreen',
                 'blue', 'cyan', 'magenta',
                 'lightblue', 'lightcyan']
    cmap = plt.cm.colors.ListedColormap(cmap_list)
    bounds = [-0.5 + i for i in range(len(cmap_list) + 1)]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    plot_grid = np.copy(grid_env.grid).astype(float)

    if astar_expanded_nodes_set:
        for r, c in astar_expanded_nodes_set:
            if (r,c) != start and (r,c) != goal: plot_grid[r,c] = 7 # lightblue
    if transformer_visited_set:
        for r, c in transformer_visited_set:
            if (r,c) != start and (r,c) != goal:
                if plot_grid[r,c] != 7 : plot_grid[r,c] = 8 # lightcyan

    if transformer_path:
        for r,c in transformer_path:
            if (r,c) != start and (r,c) != goal: plot_grid[r,c] = 5
    if astar_path:
        for r,c in astar_path:
            if (r,c) != start and (r,c) != goal:
                if plot_grid[r,c] == 5:
                    plot_grid[r,c] = 6
                else:
                    plot_grid[r,c] = 4

    if start: plot_grid[start] = 3
    if goal:  plot_grid[goal] = 2

    fig_w = max(8, grid_w / 10 if grid_w > 0 else 8) # Avoid division by zero if grid_w is 0
    fig_h = max(8, grid_h / 10 if grid_h > 0 else 8) # Avoid division by zero if grid_h is 0
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(plot_grid, cmap=cmap, norm=norm, interpolation='nearest')
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, grid_w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, grid_h, 1), minor=True)
    ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", size=0)

    tick_step_w = max(1, grid_w // 15 if grid_w > 15 else 1)
    tick_step_h = max(1, grid_h // 15 if grid_h > 15 else 1)
    if grid_w > 0 : ax.set_xticks(np.arange(0, grid_w, tick_step_w))
    if grid_h > 0 : ax.set_yticks(np.arange(0, grid_h, tick_step_h))
    plt.title(title, fontsize=10)
    # DO NOT CALL plt.show() in non-interactive script
    # plt.savefig() will be called in run_demo_slurm.py

# --- Tensor Handling ---
def tensors_to_device(batch, device):
    """Moves a batch of tensors (or list/tuple of tensors) to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return [tensors_to_device(item, device) for item in batch]
    elif isinstance(batch, dict):
        return {k: tensors_to_device(v, device) for k, v in batch.items()}
    else:
        return batch # Keep non-tensor elements as they are
