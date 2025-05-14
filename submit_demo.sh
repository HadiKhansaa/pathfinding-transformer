#!/bin/bash
#SBATCH --job-name=pathfinder_demo
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100d32q:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1      # Adjust as needed
#SBATCH --mem=8G               # Adjust as needed
#SBATCH --time=00:30:00        # 15 minutes should be plenty for one demo
#SBATCH --output=logs/demo_slurm_%j.log
#SBATCH --gres=gpu:1           # If your Transformer inference is significantly faster on GPU

echo "Setting up environment..."
# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate
# Example: source ~/my_venvs/pathenv/bin/activate

# Or load necessary modules if required on your cluster
# module load python/3.9 cuda/11.x cudnn/8.x ... # Ensure PyTorch compatible versions

echo "Starting demo run..."

# --- Configuration ---
MAP_FILENAME="/home/hma153/Datasets/a_star_maps/dao-map/arena2.map" # Make sure this file is in the current directory or provide full path
MODEL_FILENAME="trained_models/model_best.pth" # Make sure this file is accessible
OUTPUT_PLOT_FILENAME="demo_output/demo_plot_${SLURM_JOB_ID}.png"

# Optional: Define specific start/goal for this demo run
# START_R=10
# START_C=10
# GOAL_R=40
# GOAL_C=40

mkdir -p logs
mkdir -p demo_output # Directory to save the plot

CMD_ARGS="--map_file ${MAP_FILENAME} \
          --model_path ${MODEL_FILENAME} \
          --output_plot_file ${OUTPUT_PLOT_FILENAME}"

# Add start/goal to command if defined
# if [ ! -z "$START_R" ]; then
#    CMD_ARGS="${CMD_ARGS} --start_row ${START_R} --start_col ${START_C} --goal_row ${GOAL_R} --goal_col ${GOAL_C}"
# fi

# Add other args if needed
# CMD_ARGS="${CMD_ARGS} --max_tf_steps 2000"


echo "Running Python demo script..."
python run_demo_slurm.py ${CMD_ARGS}

echo "Demo script finished. Plot saved to ${OUTPUT_PLOT_FILENAME}"
echo "Check Slurm log: logs/demo_slurm_${SLURM_JOB_ID}.log"
