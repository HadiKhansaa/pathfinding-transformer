import torch
from torch.utils.data import Dataset
import numpy as np

import config
from utils import extract_local_patch

class PathfindingPatchDataset(Dataset):
    def __init__(self, trajectory_data, patch_size, target_grid_size):
        """
        Args:
            trajectory_data (list): List of (state, action_idx) tuples.
                                     state is (grid_array, current_pos, goal_pos).
            patch_size (int): The side length of the local patch.
            target_grid_size (int): The maximum grid size the coordinates refer to.
        """
        self.trajectory_data = trajectory_data
        self.patch_size = patch_size
        self.target_grid_size = target_grid_size # Used for coordinate clamping
        self.max_coord_value = target_grid_size - 1

    def __len__(self):
        return len(self.trajectory_data)

    def __getitem__(self, idx):
        state, action_idx = self.trajectory_data[idx]
        grid_array, current_pos, goal_pos = state

        # Ensure grid_array is numpy
        if not isinstance(grid_array, np.ndarray):
            grid_array = np.array(grid_array)

        # Extract the local patch centered around the current position
        patch_array = extract_local_patch(
            grid_array,
            current_pos[0],
            current_pos[1],
            self.patch_size
        )

        # Flatten the patch
        flat_patch = patch_array.flatten()

        # Convert to tensors
        patch_tensor = torch.tensor(flat_patch, dtype=torch.long)

        # Clamp coordinates to ensure they are valid indices for embeddings
        current_pos_clamped = torch.tensor(current_pos, dtype=torch.long).clamp(0, self.max_coord_value)
        goal_pos_clamped = torch.tensor(goal_pos, dtype=torch.long).clamp(0, self.max_coord_value)

        action_tensor = torch.tensor(action_idx, dtype=torch.long)

        # Return: patch, current_pos, goal_pos, action
        return patch_tensor, current_pos_clamped, goal_pos_clamped, action_tensor
