#!/bin/bash
#SBATCH --job-name=eval_pathfinder
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00      # Adjust time limit as needed
#SBATCH --output=logs/eval_%j.log # Dynamic log name

echo "Setting up environment..."
# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Or load necessary modules if required on your cluster
# module load python/3.9 cuda/11.x cudnn/8.x ...

echo "Starting evaluation run..."

# --- Configuration for the MODEL BEING EVALUATED ---
# Option 1: If you always evaluate the 'best' model from the default training run
# MODEL_FILE_TO_EVALUATE="trained_models/model_best.pth"
# TRAINED_MODEL_PATCH_SIZE=15   # Example: if you trained with patch_size 15
# TRAINED_MODEL_COORD_SIZE=512  # Example: if you trained with model_coord_vocab_size 512

# Option 2: If you have multiple model versions (e.g., from different hyperparams)
# Specify the exact model file and its training parameters
MODEL_DIR="trained_models/bigger_model_run1" # Example custom model directory
MODEL_FILE_TO_EVALUATE="${MODEL_DIR}/model_best.pth"
TRAINED_MODEL_PATCH_SIZE=15    # The patch size used to train THIS specific model
TRAINED_MODEL_COORD_SIZE=512   # The model_coord_vocab_size used for THIS model

# --- Configuration for the EVALUATION SCENARIO ---
EVAL_ON_GAME_GRIDS=true # Set to true to use game grids, false for generated
NUM_SCENARIOS_PER_GAME_MAP=3 # If EVAL_ON_GAME_GRIDS is true
GAME_GRIDS_DIR_PATH="~/Datasets/a_star_maps/"

# For generated grids (if EVAL_ON_GAME_GRIDS is false)
EVAL_ON_MAZE_GENERATED=false # If true, use mazes; if false use random obstacles
NUM_GENERATED_CASES=200
EVAL_GRID_SIZE_GENERATED=100 # Size of randomly generated grids for testing

# --- Output Directory ---
# Create a unique results directory based on model and scenario
MODEL_NAME_TAG=$(basename "${MODEL_DIR:-trained_models}") # Get a tag from model dir or default
if [ "$EVAL_ON_GAME_GRIDS" = true ]; then
    SCENARIO_TAG="game_maps_s${NUM_SCENARIOS_PER_GAME_MAP}"
else
    if [ "$EVAL_ON_MAZE_GENERATED" = true ]; then
        SCENARIO_TAG="maze_g${EVAL_GRID_SIZE_GENERATED}_n${NUM_GENERATED_CASES}"
    else
        SCENARIO_TAG="random_g${EVAL_GRID_SIZE_GENERATED}_n${NUM_GENERATED_CASES}"
    fi
fi
RESULTS_SUBDIR="evaluation_results/${MODEL_NAME_TAG}_on_${SCENARIO_TAG}"

mkdir -p "${RESULTS_SUBDIR}"
mkdir -p logs # Ensure logs directory exists

echo "Evaluating model: ${MODEL_FILE_TO_EVALUATE}"
echo "Trained with patch size: ${TRAINED_MODEL_PATCH_SIZE}, coord size: ${TRAINED_MODEL_COORD_SIZE}"
echo "Evaluation scenario tag: ${SCENARIO_TAG}"
echo "Results will be saved in: ${RESULTS_SUBDIR}"


# --- Construct the evaluate.py command ---
CMD="python evaluate.py \
    --model_path \"${MODEL_FILE_TO_EVALUATE}\" \
    --model_patch_size ${TRAINED_MODEL_PATCH_SIZE} \
    --model_coord_size ${TRAINED_MODEL_COORD_SIZE} \
    --results_dir \"${RESULTS_SUBDIR}\" "

if [ "$EVAL_ON_GAME_GRIDS" = true ]; then
    CMD+="--use-game-grids \
          --game_grids_dir \"${GAME_GRIDS_DIR_PATH}\" \
          --num_scenarios_per_map ${NUM_SCENARIOS_PER_GAME_MAP} "
else
    CMD+="--num_cases ${NUM_GENERATED_CASES} \
          --eval_grid_size ${EVAL_GRID_SIZE_GENERATED} "
    if [ "$EVAL_ON_MAZE_GENERATED" = true ]; then
        CMD+="--use-maze-grids "
    else
        CMD+="--use-random-grids "
        # You might want to add --obstacle_density if it's relevant for random grids
        # CMD+="--obstacle_density 0.3 "
    fi
fi

# Add other common args like --max_steps if needed
# CMD+="--max_steps 5000 "

echo "Executing: ${CMD}"
eval "${CMD}" # Use eval to correctly handle quotes in paths

echo "Evaluation script finished."
