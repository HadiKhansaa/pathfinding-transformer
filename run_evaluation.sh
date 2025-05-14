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


# Ensure results and log directories exist
mkdir -p evaluation_results/game_maps_latest_2
mkdir -p logs

# --- IMPORTANT: Set these according to your TRAINED model ---
MODEL_WAS_TRAINED_WITH_PATCH_SIZE=15
MODEL_WAS_TRAINED_WITH_COORD_SIZE=1261 # e.g., if your training TARGET_GRID_SIZE was 100
# ---

# Run evaluation script
# Make sure patch_size and grid_size match the model being evaluated
python evaluate.py \
    --use-game-grids \
    --game_grids_dir "~/Datasets/a_star_maps/" \
    --num_scenarios_per_map 2 \
    --model_path "trained_models/model_best.pth" \
    --model_patch_size ${MODEL_WAS_TRAINED_WITH_PATCH_SIZE} \
    --model_coord_size ${MODEL_WAS_TRAINED_WITH_COORD_SIZE} \
    --results_dir "evaluation_results/game_maps_latest_2" \
    # --max_steps 10000 # Optional: Set a higher default for large game maps
