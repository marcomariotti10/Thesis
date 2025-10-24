"""
Diffusion Model Inference Visualization

This module provides comprehensive visualization and evaluation tools for diffusion-based
neural networks trained on LiDAR data prediction tasks. It implements DDIM sampling,
model evaluation metrics, and interactive visualization capabilities.

Key Features:
- DDIM (Deterministic Diffusion Implicit Models) sampling for inference
- Noise schedule management and timestep handling
- Model evaluation with loss computation across test chunks
- Interactive prediction visualization with ground truth comparison
- Support for distributed training and multi-GPU configurations
- Configurable sampling parameters and visualization options

The diffusion process uses a learned denoising network to iteratively refine predictions
from pure noise to coherent future LiDAR occupancy maps, enabling probabilistic
prediction of dynamic environments.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import gc
import torch.distributed as dist
import argparse
from torchsummary import summary

def visualize_prediction(pred, gt, map):
    """
    Visualize diffusion model predictions alongside ground truth with LiDAR background.
    
    This function creates a side-by-side comparison visualization showing the model's
    predicted future occupancy map and the corresponding ground truth, both overlaid
    on the original LiDAR grid map for spatial context.
    
    Args:
        pred (np.ndarray): Predicted binary occupancy map, shape (400, 400)
                          Values represent predicted object presence (0-1 range)
        gt (np.ndarray): Ground truth binary occupancy map, shape (400, 400)
                        Values represent actual object presence (0-1 range)  
        map (np.ndarray): Background LiDAR grid map, shape (400, 400)
                         Provides spatial context with height/occupancy information
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    
    # Left panel: Prediction overlay
    ax[0].imshow(map, cmap='gray', alpha=0.5)      # Background LiDAR map
    ax[0].imshow(pred, cmap='jet', alpha=0.5)      # Prediction overlay
    ax[0].set_title('Overlay of Original and Prediction Grid Maps')

    # Right panel: Ground truth overlay  
    ax[1].imshow(map, cmap='gray', alpha=0.5)      # Background LiDAR map
    ax[1].imshow(gt, cmap='jet', alpha=0.5)        # Ground truth overlay
    ax[1].set_title('Overlay of Original and Ground Truth Grid Maps')
    
    plt.show()
    plt.show()  # Note: Duplicate show() call - might be intentional for certain display setups

def model_preparation(model_name, model_type):
    """
    Prepare and load a trained diffusion model for inference and evaluation.
    
    This function handles the complete model setup process for diffusion models,
    including hardware detection, model instantiation, checkpoint loading, and
    multi-GPU configuration. It also displays model architecture information.
    
    Args:
        model_name (str): Name of the saved model file (without .pth extension)
                         Used to locate the model checkpoint in MODEL_DIR
        model_type (class): Model class constructor (e.g., DiffusionModel, BigUNet)
                           Used to instantiate the model architecture
    
    Returns:
        tuple: (model, device)
            - model: Loaded PyTorch model in evaluation mode, ready for inference
                    Wrapped in DataParallel if multiple GPUs available
            - device: PyTorch device object (cuda or cpu) for tensor operations
    """
    # Display CUDA and hardware capabilities
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Get current CUDA device information
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    # Instantiate model architecture
    model = model_type()
    
    # Construct full path to model checkpoint
    model_path = MODEL_DIR
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
    
    # Configure device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Extract state dictionary (handle different checkpoint formats)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Clean state dictionary by removing 'module.' prefix if present
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load cleaned state dictionary into model
    model.load_state_dict(new_state_dict)

    # Move model to appropriate device
    model = model.to(device)

    # Configure multi-GPU setup if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    # Display model architecture summary (handles both single and multi-GPU models)
    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=[(NUMBER_FRAMES_INPUT, 400, 400), (1, 400, 400), (1,400,400)])
    else:
        summary(model, input_size=[(NUMBER_FRAMES_INPUT, 400, 400), (1, 400, 400), (1,400,400)])
    
    # Set model to evaluation mode for inference
    model.eval()

    return model, device


def show_predictions(model, device):
    """
    Interactive visualization of diffusion model predictions using DDIM sampling.
    
    This function demonstrates the complete diffusion inference process by starting
    from pure noise and iteratively denoising to produce coherent future occupancy
    predictions. It uses DDIM (Denoising Diffusion Implicit Models) for deterministic
    sampling and provides interactive visualization.
    
    Args:
        model: PyTorch diffusion model for inference
               Expected to predict noise from (past_frames, noisy_input, timestep)
        device: PyTorch device for tensor operations (cuda or cpu)
    
    Process:
        1. Load noise schedules (beta, alpha, alpha_cumprod) for diffusion math
        2. Configure DDIM sampling parameters (T timesteps, deterministic)
        3. For each test data chunk:
           - Load single samples for detailed visualization
           - Initialize prediction from pure Gaussian noise
           - Run reverse diffusion process (T steps)
           - Apply DDIM update equations at each timestep
           - Visualize final denoised prediction vs ground truth
    
    DDIM Sampling Algorithm:
        - Starts from x_T ~ N(0, I) (pure noise)
        - For t = T-1, T-2, ..., 0:
          * Predict noise: ε̂ = model(x_t, past_frames, t)
          * Estimate x_0: x̂_0 = (x_t - √(1-ᾱ_t) * ε̂) / √(ᾱ_t)
          * Update: x_{t-1} = √(ᾱ_{t-1}) * x̂_0 + √(1-ᾱ_{t-1}) * ε̂
        - Final result: x_0 (denoised prediction)
    """
    # Load complete noise schedule for diffusion mathematics
    beta, alpha, alpha_cumprod = get_noise_schedule()

    batch_size = 1
    sigmoid = torch.nn.Sigmoid()

    for chunk_id in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        test_loader = load_dataset('test', chunk_id, device, batch_size)

        model.eval()
        with torch.no_grad():
            for data in test_loader:
                predictions = []
                grid_maps = []
                vertices = []

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

                # Convert final logits to probabilities and (optionally) binarize
                x_t = sigmoid(x_recon)
                threshold = 0.5
                x_t = (x_t > threshold).float()

                # Collect predictions
                predictions.append(x_t)  # or x_bin
                predictions = torch.cat(predictions).cpu().numpy()

                # For visualization, keep your original mapping:
                grid_maps = data[0].cpu().numpy()   # inputs
                vertices  = data[1].cpu().numpy()   # targets

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                vertices = vertices.reshape(-1, 400, 400)

                for i in range(batch_size):
                    visualize_prediction(predictions[i], vertices[i], grid_maps[i])

if __name__ == '__main__':
    """
    Main execution block for diffusion model inference visualization.
    
    This script provides a command-line interface for visualizing trained diffusion
    models on test data. It supports both quantitative evaluation and qualitative
    visualization using DDIM sampling for deterministic inference.
    
    Command Line Arguments:
        --model_type: Model architecture class name (default: BigUNet)
                     Must match a class defined in the global namespace
        --model_name: Saved model filename without extension
                     Used to locate the checkpoint file in MODEL_DIR
    
    Execution Flow:
        1. Parse command line arguments for model specification
        2. Setup distributed training environment for multi-GPU support
        3. Initialize PyTorch distributed process group (NCCL backend)
        4. Load and prepare the specified diffusion model
        5. Run either evaluation metrics or prediction visualization
        6. Handle keyboard interrupts gracefully with resource cleanup
    """
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Train and test a neural network model.')
    parser.add_argument('--model_type', type=str, default='DiffusionModel', 
                       help='Type of model to use')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Name of the model to load')
    args = parser.parse_args()

    # Resolve model class from global namespace
    model_type = globals()[args.model_type]
    model_name = args.model_name

    # Configure distributed training environment for single-node multi-GPU setup
    os.environ['MASTER_ADDR'] = 'localhost'  # Master node address
    os.environ['MASTER_PORT'] = '29500'      # Communication port (adjust if needed)
    os.environ['RANK'] = '0'                 # Process rank (single process)
    os.environ['WORLD_SIZE'] = '1'           # Total number of processes
    os.environ['LOCAL_RANK'] = '0'           # Local rank for single-GPU

    # Initialize distributed process group for multi-GPU coordination
    dist.init_process_group(backend='nccl')  # Use NCCL for GPU communication
    
    # Load and prepare model for inference
    model, device = model_preparation(model_name, model_type)

    try:

        # Run interactive prediction visualization with DDIM sampling
        show_predictions(model, device)
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        # Cleanup distributed resources
        torch.distributed.destroy_process_group()
