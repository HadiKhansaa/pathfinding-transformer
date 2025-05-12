import torch
import torch.nn.functional as F
import numpy as np

import config
from model import create_model # Or directly PathfindingTransformer if needed
from utils import action_to_delta, extract_local_patch

class TransformerPlanner:
    def __init__(self, model_path, patch_size, target_grid_size, device):
        """
        Args:
            model_path (str): Path to the trained model state_dict (.pth file).
            patch_size (int): Patch size the model was trained with.
            target_grid_size (int): Grid size the model expects for coordinates.
            device: PyTorch device.
        """
        self.patch_size = patch_size
        self.target_grid_size = target_grid_size
        self.max_coord_value = target_grid_size - 1
        self.device = device

        print(f"Initializing planner with patch size {patch_size} and target grid {target_grid_size}x{target_grid_size}")

        # Create a model instance with the architecture used during training
        # Note: Hyperparameters like embed_dim etc., must match the saved model!
        # Ideally, load these from the checkpoint args if saved. For now, use config.
        # TODO: Load model hyperparameters from checkpoint args if possible for robustness
        self.model = create_model() # Assumes config matches saved model's arch
        try:
            # Load the state dict
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.to(device)
            self.model.eval()
            print(f"Transformer model loaded successfully from {model_path}")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {model_path}. Planner will not work.")
            self.model = None
        except Exception as e:
            print(f"ERROR: Could not load model state_dict from {model_path}. Error: {e}")
            print("Ensure the model architecture in model.py matches the saved checkpoint.")
            self.model = None

    def plan_path(self, grid_env, start_node, goal_node, max_steps=None):
        """
        Plans a path using the loaded Transformer model.

        Args:
            grid_env (GridEnvironment): The environment instance.
            start_node (tuple): Start coordinates (r, c).
            goal_node (tuple): Goal coordinates (r, c).
            max_steps (int, optional): Maximum planning steps. Defaults to config.MAX_PLANNER_STEPS.

        Returns:
            tuple: (path, steps_taken, status)
                path (list | None): List of (r, c) tuples, or None if planning fails.
                steps_taken (int): Number of steps the planner took.
                status (str): "success", "failure_stuck", "failure_max_steps",
                              "failure_invalid_start_goal", "failure_model_not_loaded".
        """
        if self.model is None:
            print("Transformer model not loaded. Cannot plan path.")
            return None, 0, "failure_model_not_loaded"

        if max_steps is None:
            max_steps = config.MAX_PLANNER_STEPS

        if grid_env.is_obstacle(start_node[0], start_node[1]) or \
           grid_env.is_obstacle(goal_node[0], goal_node[1]) or \
           not grid_env.is_valid(start_node[0], start_node[1]) or \
           not grid_env.is_valid(goal_node[0], goal_node[1]):
             # Check validity against actual grid_env size
            print(f"Warning: Invalid start {start_node} or goal {goal_node} for grid size {grid_env.size}.")
            return None, 0, "failure_invalid_start_goal"


        current_pos = start_node
        path = [current_pos]
        steps_taken = 0
        grid_array = grid_env.grid # Get the grid numpy array

        # Pre-process goal tensor (remains constant)
        # Clamp goal coordinates based on the model's expected max coordinate
        goal_pos_clamped = torch.tensor([goal_node], dtype=torch.long).clamp(0, self.max_coord_value).to(self.device)

        visited_for_cycle_detection = {current_pos}

        with torch.no_grad():
            for step in range(max_steps):
                if current_pos == goal_node:
                    return path, steps_taken, "success"

                # 1. Extract Local Patch
                patch_array = extract_local_patch(grid_array, current_pos[0], current_pos[1], self.patch_size)
                flat_patch = patch_array.flatten()
                patch_tensor = torch.tensor(flat_patch, dtype=torch.long).unsqueeze(0).to(self.device) # Add batch dim

                # 2. Prepare Current Position Tensor
                current_pos_clamped = torch.tensor([current_pos], dtype=torch.long).clamp(0, self.max_coord_value).to(self.device)

                # 3. Model Inference
                action_logits = self.model(patch_tensor, current_pos_clamped, goal_pos_clamped)
                action_probs = F.softmax(action_logits, dim=-1)

                # 4. Action Selection (Greedy with cycle/obstacle check)
                sorted_actions = torch.argsort(action_probs, dim=-1, descending=True).squeeze().tolist()
                if not isinstance(sorted_actions, list): # Handle case of single output
                     sorted_actions = [sorted_actions]

                moved = False
                for action_idx in sorted_actions:
                    dr, dc = action_to_delta(action_idx)
                    next_r, next_c = current_pos[0] + dr, current_pos[1] + dc

                    # Check validity within the actual grid environment
                    if grid_env.is_valid(next_r, next_c) and \
                       not grid_env.is_obstacle(next_r, next_c) and \
                       (next_r, next_c) not in visited_for_cycle_detection:
                        # Move is valid
                        current_pos = (next_r, next_c)
                        path.append(current_pos)
                        steps_taken += 1
                        visited_for_cycle_detection.add(current_pos) # Add to visited set
                        moved = True
                        break # Action taken, proceed to next step

                if not moved:
                    # print(f"Planner stuck at {current_pos} after {steps_taken} steps. Goal: {goal_node}.")
                    # print(f" Obstacles around: N:{grid_env.is_obstacle(current_pos[0]-1, current_pos[1])}, E:{grid_env.is_obstacle(current_pos[0], current_pos[1]+1)}, S:{grid_env.is_obstacle(current_pos[0]+1, current_pos[1])}, W:{grid_env.is_obstacle(current_pos[0], current_pos[1]-1)}")
                    # print(f" Probabilities: {action_probs.cpu().numpy().round(3)}")
                    return path, steps_taken, "failure_stuck"

            # Loop finished
            if current_pos == goal_node: # Check again in case max_steps reached exactly on goal
                 return path, steps_taken, "success"
            else:
                 # print(f"Planner failed to reach goal {goal_node} within max steps ({max_steps}). Final pos: {current_pos}")
                 return path, steps_taken, "failure_max_steps"
