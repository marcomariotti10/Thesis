import torch
import numpy as np
import os
import math
import pickle
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, TensorDataset
from functions_for_NN import *
from constants import *


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


if __name__ == '__main__':

    number_of_chucks_testset = NUMBER_OF_CHUNCKS_TEST

    # Load model
    model_path = MODEL_DIR
    model_name = 'model_20250228_160808_loss_0.0272'
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    criterion = WeightedCustomLoss()
    model.eval()

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    summary(model, (1, 400, 400))

    predictions = []
   
    for i in range(number_of_chucks_testset): #type: ignore
        
        print(f"\nChunck number {i+1} of {number_of_chucks_testset}")
        
        test_loader = load_dataset('test', i, device, 1)

        print("\nLenght of the datasets:", len(test_loader))        
        
        # Make predictions
        with torch.no_grad():
            for data in test_loader:
                predictions = []
                grid_maps = []
                vertices = []
                inputs, target = data
                outputs = model(inputs)
                predictions.append(outputs)
                predictions = torch.cat(predictions).cpu().numpy()
                print("Predictions Shape:", predictions.shape)
                
                grid_maps = data[0].cpu().numpy()
                vertices = data[1].cpu().numpy()

                # Now covariate_data and label_data are numpy arrays containing all the elements
                print("Covariate data shape:", grid_maps.shape)
                print("Label data shape:", vertices.shape)

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                vertices = vertices.reshape(-1, 400, 400)
                
                for i in range(len(grid_maps)):
                    visualize_prediction(predictions[i], vertices[i], grid_maps[i])



    for i in range(predictions.shape[0]):
        
        pred = predictions[i].squeeze()
        gt = vertices[i].squeeze()
        map = grid_maps[i].squeeze()

        print("\nShapes:", pred.shape, gt.shape, map.shape)
      
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(map, cmap='gray', alpha=0.5)
        ax[0].imshow(pred, cmap='jet', alpha=0.5)
        ax[0].set_title('Overlay of Original and Prediction Grid Maps')

        ax[1].imshow(map, cmap='gray', alpha=0.5)
        ax[1].imshow(gt, cmap='jet', alpha=0.5)
        ax[1].set_title('Overlay of Original and Ground Truth Grid Maps')
        plt.show()

