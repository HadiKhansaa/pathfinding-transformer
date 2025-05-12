import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import config
from model import create_model
from dataset import PathfindingPatchDataset
from data_generation import load_data
from utils import tensors_to_device

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Pathfinding Transformer")
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--patch_size', type=int, default=config.PATCH_SIZE, help='Size of local patch')
    parser.add_argument('--grid_size', type=int, default=config.TARGET_GRID_SIZE, help='Grid size for data/coords')
    parser.add_argument('--embed_dim', type=int, default=config.EMBED_DIM, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=config.NUM_HEADS, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=config.NUM_LAYERS, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=config.D_FF, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=config.DROPOUT, help='Dropout rate')
    parser.add_argument('--data_dir', type=str, default=config.EXPERT_DATA_DIR, help='Directory for expert data')
    parser.add_argument('--model_dir', type=str, default=config.MODEL_SAVE_DIR, help='Directory to save models')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')
    parser.add_argument('--no_validate', action='store_true', help='Skip validation loop during training')
     # Add more arguments as needed for HPT or specific runs
    return parser.parse_args()

# --- Training Loop ---
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip_val):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)

    for batch in progress_bar:
        # Move batch to device
        patch, current_pos, goal_pos, actions = tensors_to_device(batch, device)

        optimizer.zero_grad()
        action_logits = model(patch, current_pos, goal_pos)
        loss = criterion(action_logits, actions)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)

# --- Validation Loop ---
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    progress_bar = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            patch, current_pos, goal_pos, actions = tensors_to_device(batch, device)

            action_logits = model(patch, current_pos, goal_pos)
            loss = criterion(action_logits, actions)
            total_loss += loss.item()

            _, predicted_actions = torch.max(action_logits, 1)
            correct_predictions += (predicted_actions == actions).sum().item()
            total_samples += actions.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy

# --- Main Training Function ---
def main(args):
    print("Starting Training Process...")
    print(f'Arguments: {args}')

    # --- Setup ---
    device = config.DEVICE
    os.makedirs(args.model_dir, exist_ok=True)

    # --- Load Data ---
    print("Loading data...")
    train_data_path = os.path.join(args.data_dir, os.path.basename(config.TRAIN_DATA_FILE)) # Construct path
    val_data_path = os.path.join(args.data_dir, os.path.basename(config.VAL_DATA_FILE))

    train_trajectory_data = load_data(train_data_path)
    val_trajectory_data = load_data(val_data_path)

    if train_trajectory_data is None or val_trajectory_data is None:
        print("Error: Could not load training or validation data. Exiting.")
        print(f"Looked for: {train_data_path}, {val_data_path}")
        print(f"Generate data first using data_generation.py or check paths.")
        return

    # --- Create Datasets and DataLoaders ---
    # Use args for patch_size and grid_size passed to dataset
    train_dataset = PathfindingPatchDataset(train_trajectory_data, args.patch_size, args.grid_size)
    val_dataset = PathfindingPatchDataset(val_trajectory_data, args.patch_size, args.grid_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # --- Initialize Model, Optimizer, Criterion ---
    # Update config temporarily if args differ (cleaner way might involve passing dict)
    config.EMBED_DIM = args.embed_dim
    config.NUM_HEADS = args.num_heads
    config.NUM_LAYERS = args.num_layers
    config.D_FF = args.d_ff
    config.DROPOUT = args.dropout
    config.PATCH_SIZE = args.patch_size
    config.COORD_VOCAB_SIZE = args.grid_size
    config.MODEL_MAX_SEQ_LEN = (args.patch_size * args.patch_size) + 2

    model = create_model() # Uses updated config values
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    # Optional: Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # --- Resume Training ---
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses, val_accuracies = [], [], []

    checkpoint_path = os.path.join(args.model_dir, os.path.basename(config.CHECKPOINT_FILE))
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf')) # Load best loss if saved
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Resumed from Epoch {start_epoch}, Best Val Loss: {best_val_loss:.4f}")

    # --- Training Loop ---
    print(f"Starting training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, args.epochs):
        epoch_num = epoch + 1
        print(f"\n--- Epoch {epoch_num}/{args.epochs} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config.GRADIENT_CLIP_VAL)
        train_losses.append(train_loss)
        print(f"Epoch {epoch_num} Training Loss: {train_loss:.4f}")

        if not args.no_validate:
            val_loss, val_accuracy = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch_num} Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

            # LR Scheduler Step
            if scheduler:
                scheduler.step(val_loss)

            # --- Save Checkpoint ---
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
                'args': args # Save args used for this run
            }
            if scheduler:
                 checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(checkpoint_data, checkpoint_path)
            # print(f"Checkpoint saved to {checkpoint_path}") # Optional: print every time

            # --- Save Best Model ---
            if val_loss < best_val_loss:
                print(f"Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving best model...")
                best_val_loss = val_loss
                best_model_path = os.path.join(args.model_dir, os.path.basename(config.BEST_MODEL_FILE))
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")

            # --- HPT Integration Point ---
            # If using Optuna, report val_loss or val_accuracy here:
            # trial.report(val_loss, epoch)
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
        else:
             # If skipping validation, save checkpoint periodically or just at the end
             if epoch % 5 == 0 or epoch == args.epochs - 1:
                 torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict()}, checkpoint_path)


    print("\nTraining finished.")

    # --- Plotting ---
    if not args.no_validate and train_losses and val_losses:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_filename = os.path.join(args.model_dir, "training_plots.png")
        plt.savefig(plot_filename)
        print(f"Training plots saved to {plot_filename}")
        # plt.show() # Comment out for non-interactive Slurm jobs

if __name__ == "__main__":
    args = parse_args()
    main(args)
