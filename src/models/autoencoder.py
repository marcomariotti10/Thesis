"""
Autoencoder Model Training Script

This script implements a multi-head autoencoder model for sequential data prediction.
The model takes input sequences and produces multiple future time step predictions,
with support for distributed training on multiple GPUs.

Key Features:
- Multi-head output architecture for predicting multiple future time steps
- Distributed training support with DataParallel
- Early stopping and learning rate scheduling
- Chunk-based training for handling large datasets
- Model checkpointing with best model saving
"""

import sys
import os
# Add parent directory to path for config imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torchsummary import summary
import gc
import random
import torch.distributed as dist
import argparse

def define_models(model_type):
    """
    Initialize and configure the autoencoder model for training.
    
    This function creates the model instance, sets up GPU configuration,
    applies DataParallel for multi-GPU training, and initializes weights.
    
    Args:
        model_type (class): The model class to instantiate (e.g., SmallModel, MediumModel, LargeModel)
        
    Returns:
        tuple: (model, device) where:
            - model: The initialized and configured PyTorch model
            - device: The device (cuda/cpu) where the model is located
    """
    # Create model instance from the provided class
    model = model_type()

    # Display CUDA availability information for debugging
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Configure multi-GPU training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        # Wrap model with DataParallel for distributed training across GPUs
        model = nn.DataParallel(model)

    # Move model to appropriate device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Display current CUDA device information
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    # Print model summary with appropriate input size
    # Handle DataParallel wrapper for summary display
    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=(NUMBER_FRAMES_INPUT, 400, 400))
    else:
        summary(model, input_size=(NUMBER_FRAMES_INPUT, 400, 400))

    # Apply custom weight initialization to all model layers
    model.apply(initialize_weights)

    return model, device

def train(model, device):
    """
    Train the multi-head autoencoder model with chunk-based data loading.
    
    This function implements the complete training loop including:
    - Multi-head loss calculation across future time steps
    - Chunk-based data loading for memory efficiency
    - Early stopping and learning rate scheduling
    - Model checkpointing with best model saving
    - Validation evaluation across multiple chunks
    
    The model predicts multiple future time steps, and the loss is averaged
    across all prediction heads.
    
    Args:
        model (nn.Module): The initialized autoencoder model
        device (torch.device): The device to run training on (cuda/cpu)
    """
    
    # Initialize optimizer with Adam algorithm
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE_LR_SCHEDULER)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=PATIENCE_EARLY_STOPPING, min_delta=0.0001)
    
    # Binary Cross Entropy loss with logits for multi-head outputs
    # pos_weight can be adjusted to handle class imbalance
    pos_weight = torch.tensor([1], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
   
    # Training configuration parameters from config file
    early_stopping_triggered = False
    number_of_chuncks = NUMBER_OF_CHUNCKS_TRAIN      # Number of training data chunks
    num_total_epochs = NUMBER_EPOCHS                  # Total training epochs
    number_of_chuncks_val = NUMBER_OF_CHUNCKS_TEST   # Number of validation chunks
    batch_size = BATCH_SIZE                          # Batch size for training

    # Ensure model directory exists and set up model checkpointing
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_loss = float('inf')  # Track best validation loss for model saving
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')

    # Main training loop across epochs
    for j in range(num_total_epochs):
        if early_stopping_triggered:
            break

        # Randomize chunk order for each epoch to improve generalization
        random.seed(SEED + j)
        train_order = random.sample(range(number_of_chuncks), number_of_chuncks)
        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")

        # Training loop across data chunks
        for i in range(number_of_chuncks):  # type: ignore
            if early_stopping_triggered:
                break

            print(f"\Shard number {i+1} of {number_of_chuncks}")
            train_index = train_order[i]
            
            # Load current training chunk
            train_loader = load_dataset('train', train_index, device, batch_size)

            start = datetime.now()
            model.train()  # Set model to training mode
            train_loss = 0
            
            # Training loop within current chunk
            for data in train_loader:
                
                inputs, targets = data
                
                # Dynamically handle multiple future time steps
                num_targets = targets.shape[1]  # Get number of future time steps to predict
                targets = [
                    targets[:, i].unsqueeze(1).float() for i in range(num_targets)
                ]
                
                # Zero gradients from previous iteration
                optimizer.zero_grad()

                # Forward pass: get predictions for all future time steps
                pred = model(inputs)

                # Calculate multi-head loss: average loss across all prediction heads
                loss = 0
                for z in range(num_targets):
                    # Add loss for each future time step prediction
                    loss += criterion(pred[z], targets[z])
                loss /= num_targets  # Average loss across all heads
                                
                # Backward pass and parameter update
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Calculate average training loss for this chunk
            train_loss /= len(train_loader)

            # Validation evaluation across all validation chunks
            val_losses = []
            for k in range(number_of_chuncks_val):
                val_loader = load_dataset('val', k, device, batch_size)
                model.eval()  # Set model to evaluation mode
                val_loss = 0
                
                # Validation loop (no gradient computation)
                with torch.no_grad():
                    for data in val_loader:
                        inputs, targets = data
                        
                        # Prepare targets for multi-head evaluation
                        num_targets = targets.shape[1]  # Get number of time steps
                        targets = [
                            targets[:, i].unsqueeze(1).float() for i in range(num_targets)
                        ]

                        # Forward pass for validation
                        pred = model(inputs)

                        # Calculate validation loss across all prediction heads
                        loss = 0
                        for z in range(num_targets):
                            loss += criterion(pred[z], targets[z])
                        loss /= num_targets  # Average over all heads
    
                        val_loss += loss.item()
                        
                # Average validation loss for this chunk
                val_loss /= len(val_loader)
                val_losses.append(val_loss)

            # Calculate total validation loss across all chunks
            total_val_loss = sum(val_losses) / len(val_losses)
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {total_val_loss:.4f}        Time: {datetime.now() - start}')

            # Save model if validation loss improved
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

            # Learning rate scheduling and early stopping checks
            scheduler.step(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                early_stopping_triggered = True
                break

    # Clean up GPU memory after training
    torch.cuda.empty_cache()
    gc.collect()

    # Rename best model with timestamp and model type for final saving
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'{time}_{model_type.__name__}.pth'
    final_model_path = os.path.join(MODEL_DIR, model_name)
    
    # Move the best model to final location with proper naming
    if os.path.exists(best_model_path):
        os.rename(best_model_path, final_model_path)
        print("\n\n")
        print("TRAINING COMPLETED")
        print(f'Best model renamed to: {model_name}')


if __name__ == "__main__":
    """
    Main execution block for training autoencoder models.
    
    This script supports training different model sizes (small, medium, large)
    through command line arguments and sets up distributed training environment.
    """

    # Parse command line arguments for model configuration
    parser = argparse.ArgumentParser(description='Train a multi head neural network model.')
    parser.add_argument('--model_size', type=str, default='medium', 
                       choices=['small', 'medium', 'large'], 
                       help='Size of model to use: small, medium, or large')
    args = parser.parse_args()

    # Map command line model size arguments to actual model classes
    model_mapping = {
        'small': SmallModel,    # Lightweight model for faster training/inference
        'medium': MediumModel,  # Balanced model with moderate complexity
        'big': LargeModel       # Complex model for maximum performance
    }
    
    # Get the model class based on user selection
    model_type = model_mapping[args.model_size]
    print(f"Using {args.model_size} model: {model_type.__name__}")

    # Configure distributed training environment variables
    # These settings enable multi-GPU training coordination
    os.environ['MASTER_ADDR'] = 'localhost'  # Master node address for distributed training
    os.environ['MASTER_PORT'] = '29500'      # Communication port (must be available)

    # Single-node multi-GPU setup configuration
    os.environ['RANK'] = '0'       # Process rank (0 for single process setup)
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes in distributed training
    os.environ['LOCAL_RANK'] = '0'  # Local rank within the node (0 for single GPU)

    # Initialize PyTorch distributed process group for multi-GPU coordination
    # NCCL backend is optimized for NVIDIA GPU communication
    dist.init_process_group(backend='nccl')

    # Clean up memory and set random seed for reproducibility
    gc.collect()
    random.seed(SEED)

    # Initialize model and device configuration
    model, device = define_models(model_type)

    try:
        # Start training process
        train(model, device)

    except KeyboardInterrupt:
        # Handle graceful shutdown on user interruption (Ctrl+C)
        print("Training interrupted by user.")

    # Clean up distributed training resources
    torch.distributed.destroy_process_group()