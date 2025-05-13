import torch
import numpy as np

# --- Environment & Data ---
TARGET_GRID_SIZE = 100      # Grid size for generation and testing
OBSTACLE_DENSITY = 0.3    # Obstacle density for random grids
USE_MAZE_GENERATION = True # Whether to include mazes in data generation
NUM_ACTIONS = 8           # 8-directional movement
# Action mapping: (dr, dc) -> action_index
# 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW
ACTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
ACTION_COSTS = [1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2), 1, np.sqrt(2)]
PATCH_SIZE = 11           # Size of the local patch (must be odd)
PATCH_PADDING_VALUE = 1   # Value to use for padding patches (1=obstacle)



# --- Data Generation ---
# ...
EXPERT_DATA_DIR = "expert_data" # Directory to save/load generated data
# Filenames will now be more dynamic if game maps are included
# TRAIN_DATA_FILE = f"{EXPERT_DATA_DIR}/train_trajectories_{TARGET_GRID_SIZE}x{TARGET_GRID_SIZE}.pkl"
# VAL_DATA_FILE = f"{EXPERT_DATA_DIR}/val_trajectories_{TARGET_GRID_SIZE}x{TARGET_GRID_SIZE}.pkl"

# --- Game Map Data for Training ---
USE_GAME_MAPS_FOR_TRAINING = True       # Set to True to include game maps in training data
GAME_MAPS_DATA_DIR = "~/Datasets/a_star_maps/" # Path to the root of game map files
GAME_MAPS_TRAIN_RATIO = 0.8             # 80% of game maps for training, rest for validation
# How many A* trajectories (S/G pairs) to generate PER selected game map for training/validation
TRAJECTORIES_PER_GAME_MAP = 10

# --- Randomly Generated Data for Training ---
# These are still used if USE_GAME_MAPS_FOR_TRAINING is True, they get combined.
# If USE_GAME_MAPS_FOR_TRAINING is False, these are the only source.
USE_RANDOM_MAPS_FOR_TRAINING = True     # Whether to also generate random/maze maps
NUM_RANDOM_TRAIN_ENVS = 5000          # Number of random/maze ENVIRONMENTS for training data
NUM_RANDOM_VAL_ENVS = 500             # Number of random/maze ENVIRONMENTS for validation data
# Note: NUM_TRAIN_TRAJECTORIES and NUM_VAL_TRAJECTORIES are effectively replaced by
# (NUM_RANDOM_TRAIN_ENVS * avg_states_per_env) + (num_selected_game_maps * TRAJECTORIES_PER_GAME_MAP * avg_states_per_env)

# --- Model ---
# ...
# COORD_VOCAB_SIZE needs to be large enough for the BIGGEST game map if training on them.
# Set this manually to a safe upper bound, or determine it dynamically in data_generation.
# For now, let's assume game maps can be up to, say, 512x512 for this example.
# You MUST adjust this based on your actual largest game map dimension.
MAX_EXPECTED_GAME_MAP_DIM = 1260
# If training on game maps, the model_coord_size should reflect this.
# The 'TARGET_GRID_SIZE' is more for the randomly generated grids.
MODEL_COORD_VOCAB_SIZE = MAX_EXPECTED_GAME_MAP_DIM if USE_GAME_MAPS_FOR_TRAINING else TARGET_GRID_SIZE



# --- Model ---
# Transformer Hyperparameters (can be tuned)
EMBED_DIM = 128
NUM_HEADS = 8             # Must divide EMBED_DIM
NUM_LAYERS = 4
D_FF = 256                # Dimension of feed-forward network
DROPOUT = 0.1
COORD_VOCAB_SIZE = TARGET_GRID_SIZE # Max coordinate value + 1
GRID_VOCAB_SIZE = 2       # 0: free, 1: obstacle (potentially add padding token if needed)
# Max sequence length for transformer based on patch size + positions
MODEL_MAX_SEQ_LEN = (PATCH_SIZE * PATCH_SIZE) + 2 # +2 for current/goal pos

# --- Training ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 128           # Adjust based on GPU memory (V100 can likely handle larger)
NUM_EPOCHS = 50           # Adjust based on convergence
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_VAL = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "trained_models"
CHECKPOINT_FILE = f"{MODEL_SAVE_DIR}/checkpoint_latest.pth"
BEST_MODEL_FILE = f"{MODEL_SAVE_DIR}/model_best.pth"

# --- Evaluation ---
MAX_PLANNER_STEPS = TARGET_GRID_SIZE * TARGET_GRID_SIZE # Max steps for transformer planner
EVAL_RESULTS_DIR = "evaluation_results"

# --- Assertions ---
assert PATCH_SIZE % 2 == 1, "PATCH_SIZE must be odd"
assert EMBED_DIM % NUM_HEADS == 0, "EMBED_DIM must be divisible by NUM_HEADS"

print(f"Using device: {DEVICE}")
print(f"Target Grid Size: {TARGET_GRID_SIZE}x{TARGET_GRID_SIZE}")
print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Model Sequence Length: {MODEL_MAX_SEQ_LEN}")
print(f"Coordinate Vocab Size: {COORD_VOCAB_SIZE}")
