#!/bin/bash
#SBATCH --job-name=train_pathfinder
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --cpus-per-task=1     # Adjust based on DataLoader workers and CPU needs
#SBATCH --mem=32G            # Adjust based on dataset size and model
#SBATCH --time=06:00:00      # Adjust time limit (HH:MM:SS) - Max 2 hours requested
#SBATCH --output=logs/train_%j.log # Log file for stdout/stderr

echo "Setting up environment..."
# Activate your virtual environment
# source /path/to/your/venv/bin/activate

# Or load necessary modules
# module load python/3.9 cuda/11.x cudnn/8.x ...

echo "Starting training run..."
# Ensure model and log directories exist
mkdir -p trained_models
mkdir -p logs

# Run the training script with specific arguments
# Example: Train with default config values
python train.py \
    --epochs 20 \
    --batch_size 512 \
    --lr 1e-5 \
    --patch_size 15 \
    # --grid_size 1261 \
    --embed_dim 256 \
    --num_heads 6 \
    --num_layers 4 \
    --d_ff 512 \
    --dropout 0.2 \
    --model_coord_vocab_size 1261
    --train_data_file "combined_train_data_train.pkl" \
    --val_data_file "combined_val_data_val.pkl" \
    # --resume # Add this flag to resume from checkpoint

# Example: Train a slightly larger model
# python train.py \
#     --epochs 50 \
#     --batch_size 64 \
#     --lr 5e-5 \
#     --patch_size 15 \
#     --grid_size 100 \
#     --embed_dim 256 \
#     --num_heads 8 \
#     --num_layers 6 \
#     --d_ff 512 \
#     --dropout 0.15 \
#     --model_dir trained_models/large_model

echo "Training script finished."
