"""
Diffusion Model Training Script

This script implements a Denoising Diffusion Probabilistic Model (DDPM) for image generation
and reconstruction tasks. The model learns to reverse a diffusion process by predicting noise
at different timesteps, enabling high-quality image synthesis.

Key Features:
- DDPM implementation with configurable noise schedules
- Distributed training support with DataParallel
- DDIM sampling for deterministic inference
- Chunk-based training for memory efficiency
- Early stopping and learning rate scheduling
- Model checkpointing with best model saving

The training process follows the DDPM objective: L = E[||ε - ε̂||^2]
where ε is the true noise and ε̂ is the predicted noise.

For inference, the model uses DDIM sampling for deterministic reconstruction
from pure noise to clean images.
"""

import sys
import os
# Add parent directory to path for config imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datetime import datetime
from torchsummary import summary
import gc
import random
import torch.nn.functional as F
import torch.distributed as dist


def define_models(model_type):
    """
    Initialize and configure the diffusion model for training.
    
    This function creates the diffusion model instance, sets up GPU configuration,
    applies DataParallel for multi-GPU training, and initializes weights.
    The model takes three inputs: conditioning frames, noisy target, and timestep.
    
    Args:
        model_type (class): The diffusion model class to instantiate
        
    Returns:
        tuple: (model, device) where:
            - model: The initialized and configured PyTorch diffusion model
            - device: The device (cuda/cpu) where the model is located
    """
    # Create diffusion model instance from the provided class
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

    # Print model summary with diffusion-specific input sizes
    # Inputs: [conditioning_frames, noisy_target, timestep]
    # Handle DataParallel wrapper for summary display
    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=[(NUMBER_FRAMES_INPUT, 400, 400), (1, 400, 400), (1,400,400)])
    else:
        summary(model, input_size=[(NUMBER_FRAMES_INPUT, 400, 400), (1, 400, 400), (1,400,400)])

    # Apply custom weight initialization to all model layers
    model.apply(initialize_weights)
    
    return model, device

def train(model, device):
    """
    Train the diffusion model using the DDPM objective and DDIM sampling for validation.
    
    This function implements the complete DDPM training pipeline:
    - Forward diffusion: gradually adds noise to clean images
    - Reverse diffusion: trains model to predict noise at each timestep
    - DDIM sampling: deterministic inference for validation
    - Chunk-based training for memory efficiency
    
    Training Process:
    1. Sample random timestep t for each image in batch
    2. Add noise according to diffusion schedule: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
    3. Train model to predict ε from (x_t, t, conditioning)
    4. Optimize using MSE loss: L = ||ε - ε̂||²
    
    Validation Process:
    1. Start from pure noise x_T ~ N(0,I)
    2. Iteratively denoise using DDIM steps
    3. Evaluate reconstruction quality against ground truth
    
    Args:
        model (nn.Module): The initialized diffusion model
        device (torch.device): The device to run training on (cuda/cpu)
    """
    
    # Initialize optimizer with Adam algorithm
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=PATIENCE_LR_SCHEDULER)
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(patience=PATIENCE_EARLY_STOPPING, min_delta=0.0001)

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

    # Get diffusion noise schedule parameters
    # beta_t: noise variance at each timestep
    # alpha_t: 1 - beta_t
    # alpha_cumprod: cumulative product of alphas (ᾱ_t)
    beta_t, alpha_t, alpha_cumprod = get_noise_schedule()

    for j in range(num_total_epochs):
        if early_stopping_triggered:
            break

        random.seed(SEED + j)
        train_order = random.sample(range(number_of_chuncks), number_of_chuncks)
        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")

        for i in range(number_of_chuncks):  # type: ignore
            if early_stopping_triggered:
                break

            print(f"\Shard number {i+1} of {number_of_chuncks}")
            train_index = train_order[i]
            train_loader = load_dataset('train', train_index, device, batch_size)

            start = datetime.now()
            model.train()
            train_loss = 0.0

            for data in train_loader:
                inputs, targets = data
                # Use the selected future map, keep shape (B,1,H,W), float
                targets = targets[:, 0].unsqueeze(1).float().to(device)
                inputs = inputs.to(device)

                optimizer.zero_grad(set_to_none=True)

                # --- DDPM: sample a timestep per sample (uniform) ---
                bsz = targets.size(0)  # <— use actual batch size
                t = torch.randint(0, RANGE_TIMESTEPS, (bsz,), device=device)

                # --- Forward diffusion: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε ---
                # If you already have this encapsulated, keep your helper:
                noisy_target, noise = get_noisy_target(targets, alpha_cumprod, t)  # returns x_t and ε
                # Ensure both tensors are on the correct device
                noisy_target = noisy_target.to(device)
                noise = noise.to(device)

                # --- Normalize t if your model expects [0,1] ---
                t_tensor = t.view(-1, 1, 1, 1).expand_as(targets) / max(1, (RANGE_TIMESTEPS - 1))
                t_tensor = t_tensor.to(device)

                # --- Predict ε from (x_t, t, conditioning) ---
                predicted_noise = model(inputs, noisy_target, t_tensor)

                # --- DDPM "simple" objective: L = E[ ||ε - ε̂||^2 ] ---
                loss = F.mse_loss(predicted_noise, noise)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_losses = []
            for k in range(number_of_chuncks_val):
                val_loader = load_dataset('val', k, device, batch_size)
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        inputs, targets = data
                        targets = targets[:, 0].unsqueeze(1).float().to(device)
                        inputs = inputs.to(device)

                        bsz = targets.size(0)

                        # Preload ᾱ_t on device
                        alpha_bar = alpha_cumprod.to(device)                     # shape [T]
                        T = RANGE_TIMESTEPS

                        # Start from pure noise: x_T ~ N(0, I)
                        x = torch.randn_like(targets)                            # x_T

                        # Deterministic DDIM-like reverse process (η=0)
                        for t_int in reversed(range(T)):
                            # t as tensor for the batch
                            t = torch.full((bsz,), t_int, device=device, dtype=torch.long)

                            # normalized t if your model expects [0,1]
                            t_tensor = t.view(-1, 1, 1, 1).expand_as(targets) / max(1, (T - 1))

                            # Predict ε from current x_t
                            eps_hat = model(inputs, x, t_tensor)

                            # Compute x0 estimate: x0_hat = (x_t - sqrt(1-ᾱ_t) * ε̂) / sqrt(ᾱ_t)
                            a_bar_t = alpha_bar[t].view(-1, 1, 1, 1).clamp(min=1e-8)
                            sqrt_a_bar_t = a_bar_t.sqrt()
                            sqrt_one_minus_a_bar_t = (1.0 - a_bar_t).clamp(min=1e-8).sqrt()
                            x0_hat = (x - sqrt_one_minus_a_bar_t * eps_hat) / sqrt_a_bar_t

                            if t_int > 0:
                                # DDIM deterministic step to t-1:
                                a_bar_t_prev = alpha_bar[t - 1].view(-1, 1, 1, 1).clamp(min=1e-8)
                                sqrt_a_bar_t_prev = a_bar_t_prev.sqrt()
                                sqrt_one_minus_a_bar_t_prev = (1.0 - a_bar_t_prev).clamp(min=1e-8).sqrt()

                                # x_{t-1} = sqrt(ᾱ_{t-1}) * x0_hat + sqrt(1-ᾱ_{t-1}) * ε̂
                                x = sqrt_a_bar_t_prev * x0_hat + sqrt_one_minus_a_bar_t_prev * eps_hat
                            else:
                                # Reached t = 0; take final reconstruction
                                x_recon = x0_hat

                        # Validation loss: compare reconstructed x0 to ground-truth target
                        loss = F.mse_loss(x_recon, targets)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                val_losses.append(val_loss)

            total_val_loss = sum(val_losses) / len(val_losses)
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {total_val_loss:.4f}        Time: {datetime.now() - start}')

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

            scheduler.step(val_loss)
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                early_stopping_triggered = True
                break
    gc.collect()

    # Rename the best model file to use the proper naming convention
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'{time}_{model_type.__name__}.pth'
    final_model_path = os.path.join(MODEL_DIR, model_name)
    
    # Rename the best_model.pth to the final model name
    if os.path.exists(best_model_path):
        os.rename(best_model_path, final_model_path)
        print("\n\n")
        print("TRAINING COMPLETED")
        print(f'Best model renamed to: {model_name}')


if __name__ == "__main__":
    """
    Main execution block for training diffusion models.
    
    This script supports different diffusion model types through command line arguments
    and sets up distributed training environment for DDPM training.
    """

    # Parse command line arguments for model selection
    parser = argparse.ArgumentParser(description='Train and test a diffusion neural network model.')
    parser.add_argument('--model_type', type=str, default='DiffusionModel', 
                       help='Type of diffusion model to use (e.g., DiffusionModel)')
    args = parser.parse_args()

    # Get model class from global namespace based on argument
    model_type = globals()[args.model_type]

    # Configure distributed training environment variables
    # These settings enable multi-GPU training coordination for DDPM
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

    # Initialize diffusion model and device configuration
    model, device = define_models(model_type)

    try:
        # Start DDPM training process
        train(model, device)

    except KeyboardInterrupt:
        # Handle graceful shutdown on user interruption (Ctrl+C)
        print("Training interrupted by user.")

    # Clean up distributed training resources
    torch.distributed.destroy_process_group()