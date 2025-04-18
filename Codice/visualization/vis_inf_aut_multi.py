import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import numpy as np
import os
import math
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
import argparse
from torchsummary import summary

def visualize_prediction(pred, gt, map):
    """
    Visualize the grid map and the prediction.
    
    Parameters:
    - grid_map: numpy array of shape (400, 400)
    - prediction: numpy array of shape (400, 400)
    """
    #print("Pred shape:", pred.shape)
    #print("GT shape:", gt.shape)
    #print("Map shape:", map.shape)

    #print((pred[2].shape))

    for i in range(len(FUTURE_TARGET_RILEVATION)):
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(map, cmap='gray', alpha=0.5)
        ax[0].imshow(pred[i], cmap='jet', alpha=0.5)
        ax[0].set_title(f'Prediction {i+1}')

        ax[1].imshow(map, cmap='gray', alpha=0.5)
        ax[1].imshow(gt[i], cmap='jet', alpha=0.5)
        ax[1].set_title('Ground Truth Grid Maps')
        plt.show()
    

def model_preparation(model_name, model_type, activation_function):

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    model = model_type(activation_fn = activation_function)
    # Load model
    model_path = MODEL_DIR
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Remove 'module.' prefix from the state dict keys if it's there
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Create a new state dict without the "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    #print("Saved model keys:", new_state_dict.keys())
    #print("Current model keys:", model.state_dict().keys())

    # Now load the cleaned state dict into your model
    model.load_state_dict(new_state_dict)

    model = model.to(device)

    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))
    else:
        summary(model, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))
    
    model.eval()

    return model, device


def evaluate(model, device):
    
    test_losses = []
    
    pos_weight = torch.tensor([1.0], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    batch_size = BATCH_SIZE

    alpha_cumprod = get_noise_schedule()

    for i in range(NUMBER_OF_CHUNCKS_TEST): #type: ignore

        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('test', i, device, batch_size)

        print("\nLenght test dataset: ", len(test_loader))

        gc.collect()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()
                
                x_t = torch.full_like(targets,0, device = device)
                #x_t = torch.randn_like(targets)

                print(x_t.shape)

                t = torch.randint(5, RANGE_TIMESTEPS, (batch_size,))  # Random timestep
                noisy_target, noise = get_noisy_target(x_t,alpha_cumprod, t)
                t_tensor = t.view(-1, 1, 1, 1).expand_as(targets)  # Reshape and expand to match targets' shape
                # Normalize t_tensor to scale values between 0 and 1
                t_tensor = t_tensor / (RANGE_TIMESTEPS - 1)
                # Move t_tensor to the appropriate device (e.g., GPU or CPU)
                t_tensor = t_tensor.to(device)
                
                # Predict the noise for this timestep
                predicted_noise = model(inputs, noisy_target, t_tensor)

                pred = noisy_target-predicted_noise

                loss = F.mse_loss(predicted_noise,noise)

                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)


def show_predictions(model, device):

    batch_size = 1
    
    sigmoid = torch.nn.Sigmoid()

    for i in range(NUMBER_OF_CHUNCKS_TEST): #type: ignore
        
        print(f"\nChunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}")
        
        test_loader = load_dataset('test', i, device, batch_size)

        print("\nLenght of the datasets:", len(test_loader))        
        
        # Make predictions
        with torch.no_grad():
            for data in test_loader:
                predictions = []
                grid_maps = []
                vertices = []
                inputs, targets = data
                targets = [
                    targets[:, i].unsqueeze(1).float().to(device)
                    for i in range(len(FUTURE_TARGET_RILEVATION))
                ]

                latent = model.encoder(inputs)
                x_t = [decoder(latent) for decoder in model.decoder]

                x_t = torch.cat(x_t, dim=1)

                # Apply threshold to convert outputs to 0 or 1
                threshold = 0.4
                x_t = (x_t > threshold).float()

                # Calculate loss with true noise
                x_t = sigmoid(x_t)
                predictions.append(x_t)
                predictions = torch.cat(predictions).cpu().numpy()
                #print("Predictions Shape:", predictions.shape)
                
                grid_maps = data[0].cpu().numpy()
                vertices = data[1].cpu().numpy()

                # Now covariate_data and label_data are numpy arrays containing all the elements
                #print("Covariate data shape:", grid_maps.shape)
                #print("Label data shape:", vertices.shape)

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                vertices = vertices.reshape(-1, 400, 400)
                
                visualize_prediction(predictions, vertices, grid_maps[i])



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test a neural network model.')
    parser.add_argument('--model_type', type=str, default='MultiHeadAutoencoder', help='Type of model to use')
    parser.add_argument('--activation_function', type=str, default='ReLU', help='Activation function to apply to the model')
    parser.add_argument('--model_name', type=str, default='model_20250415_174218_loss_0.0022_MultiHeadAutoencoder', help='Name of the model to load')
    args = parser.parse_args()

    model_type = globals()[args.model_type]
    activation_function = getattr(nn, args.activation_function)  # This retrieves the activation function class
    model_name = args.model_name

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # This is the address of the master node
    os.environ['MASTER_PORT'] = '29500'     # This is the port for communication (can choose any available port)

    # Set other environment variables for single-node multi-GPU setup
    os.environ['RANK'] = '0'       # Process rank (0 for single process)
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes
    os.environ['LOCAL_RANK'] = '0'  # Local rank for single-GPU (0 for single GPU)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')  # Use NCCL for multi-GPU setups
    
    model, device = model_preparation(model_name, model_type, activation_function)

    try:
        #evaluate(model, device)

        show_predictions(model, device)
    
    except KeyboardInterrupt:
        print("\n program interrupted by user")