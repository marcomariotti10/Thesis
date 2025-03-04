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

def load_dataset(name,i,device, batch):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path_train = os.path.join(complete_name_ffcv_path, name_train)

    train_loader = Loader(complete_path_train, batch_size=batch,
    num_workers=8, order=OrderOption.QUASI_RANDOM,
    os_cache=True,
    pipelines={
        'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                    ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                    ToDevice(device, non_blocking=True)],
        'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                ToDevice(device, non_blocking=True)]
    })

    return train_loader

def visualize_prediction(pred, gt, map):
    """
    Visualize the grid map and the prediction.
    
    Parameters:
    - grid_map: numpy array of shape (400, 400)
    - prediction: numpy array of shape (400, 400)
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(map, cmap='gray', alpha=0.5)
    ax[0].imshow(pred, cmap='jet', alpha=0.5)
    ax[0].set_title('Overlay of Original and Prediction Grid Maps')

    ax[1].imshow(map, cmap='gray', alpha=0.5)
    ax[1].imshow(gt, cmap='jet', alpha=0.5)
    ax[1].set_title('Overlay of Original and Ground Truth Grid Maps')
    plt.show()
    
    plt.show()

def model_preparation(model_name, model):
    

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    # Load model
    model_path = MODEL_DIR
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    checkpoint = torch.load(model_path, map_location=device)

    # Remove 'module.' prefix from the state dict keys if it's there
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Create a new state dict without the "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Now load the cleaned state dict into your model
    model.load_state_dict(new_state_dict)

    summary(model, (1, 400, 400))
    
    model.eval()

    return model, device


def evaluate(model, device):
    
    test_losses = []
    
    criterion = torch.nn.BCEWithLogitsLoss()

    for i in range(NUMBER_OF_CHUNCKS): #type: ignore

        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('train', i, device, 32)

        print("\nLenght test dataset: ", len(test_loader))

        gc.collect()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)


def show_predictions(model, device):
    
    sigmoid = torch.nn.Sigmoid()

    for i in range(NUMBER_OF_CHUNCKS_TEST): #type: ignore
        
        print(f"\nChunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}")
        
        test_loader = load_dataset('test', i, device, 1)

        print("\nLenght of the datasets:", len(test_loader))        
        
        # Make predictions
        with torch.no_grad():
            for data in test_loader:
                predictions = []
                grid_maps = []
                vertices = []
                inputs, target = data
                target = target.float()
                outputs = model(inputs)
                outputs = sigmoid(outputs)
                predictions.append(outputs)
                predictions = torch.cat(predictions).cpu().numpy()
                print("Predictions Shape:", predictions.shape)
                
                grid_maps = data[0].cpu().numpy()
                vertices = data[1].cpu().numpy()

                # Now covariate_data and label_data are numpy arrays containing all the elements
                #print("Covariate data shape:", grid_maps.shape)
                #print("Label data shape:", vertices.shape)

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                vertices = vertices.reshape(-1, 400, 400)
                
                for i in range(len(grid_maps)):
                    visualize_prediction(predictions[i], vertices[i], grid_maps[i])



if __name__ == '__main__':

    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *

    model_name = 'model_20250301_181025_loss_0.0020_BCEwithlogitloss'

    model_type = Autoencoder_classic()
    
    model, device = model_preparation(model_name, model_type)

    evaluate(model, device)

    show_predictions(model, device)
