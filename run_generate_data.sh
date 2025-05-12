#!/bin/bash
#SBATCH --job-name=generate_path_data
#SBATCH --nodes=1
#SBATCH --nodelist=onode17
#SBATCH --partition=gpu
##SBATCH --gres=gpu:v100d32q:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  # Adjust based on data generation needs
#SBATCH --mem=32G         # Adjust based on grid size and data size
#SBATCH --time=04:00:00   # Adjust time limit (HH:MM:SS)
#SBATCH --output=logs/generate_data_%j.log # Log file for stdout/stderr

echo "Setting up environment..."
# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Or load necessary modules if required on your cluster
# module load python/3.9 cuda/11.x ...

echo "Starting data generation..."
# Ensure the expert_data directory exists
mkdir -p expert_data
mkdir -p logs # For log files

# Run the data generation script
python data_generation.py

echo "Data generation script finished."
