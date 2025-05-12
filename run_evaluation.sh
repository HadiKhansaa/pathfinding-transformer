#!/bin/bash
#SBATCH --job-name=eval_pathfinder
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1          # Evaluation might still be faster on GPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00      # Adjust time limit
#SBATCH --output=logs/eval_%j.log

echo "Setting up environment..."
# Activate your virtual environment
# source /path/to/your/venv/bin/activate

# Or load necessary modules
# module load python/3.9 cuda/11.x ...

echo "Starting evaluation run..."
# Ensure results and log directories exist
mkdir -p evaluation_results
mkdir -p logs

# Run evaluation script
# Make sure patch_size and grid_size match the model being evaluated
python evaluate.py \
    --num_cases 500 \
    --grid_size 100 \
    --patch_size 11 \
    --model_path trained_models/model_best.pth \
    --results_dir evaluation_results/run1 \
    # --use_maze # Add flag to evaluate on mazes
    # --no_maze # Add flag to evaluate on random obstacles

echo "Evaluation script finished."
