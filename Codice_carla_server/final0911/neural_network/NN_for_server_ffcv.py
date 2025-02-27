import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv
import os
import numpy as np
import cProfile
import pstats
import sys
from sklearn.model_selection import train_test_split
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from torchsummary import summary
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
import gc
from multiprocessing import Pool, set_start_method
from matplotlib.path import Path
import open3d as o3d
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import pickle
from datetime import datetime
import random
import math
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, Squeeze, ToDevice, ToTorchImage, View
from sklearn.preprocessing import MinMaxScaler
from functions_for_NN import *
from constants import *
from ffcv.reader import Reader


def load_dataset(name,i,device):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_path_train = os.path.join(FFCV_DIR, name_train)

    train_loader = Loader(complete_path_train, batch_size=32,
    num_workers=8, order=OrderOption.QUASI_RANDOM,
    os_cache=False,
    pipelines={
        'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                    ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                    ToDevice(device, non_blocking=True)],
        'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                ToDevice(device, non_blocking=True)]
    })

    return train_loader

if __name__ == "__main__":
    
    gc.collect()

    random.seed(SEED)

    # Model creation
    model = Autoencoder()
    model.apply(weights_init)
    criterion = WeightedCustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    summary(model, (1, 400, 400))

    # Parameters for training
    early_stopping_triggered = False
    number_of_chucks= NUMBER_OF_CHUNCKS
    num_total_epochs = 1
    num_epochs_for_each_chunck = 500
    number_of_chucks_testset = NUMBER_OF_CHUNCKS_TEST

    for j in range(num_total_epochs):
        
        if early_stopping_triggered:
            break

        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")
        random.seed(SEED + j)

        for i in range(number_of_chucks): #type: ignore
            
            if early_stopping_triggered:
                break
            
            print(f"\nChunck number {i+1} of {number_of_chucks}")

            name_train = f"dataset_train{i}.beton"  # Define the path where the dataset will be written
            complete_path_train = os.path.join(FFCV_DIR, name_train)

            with ThreadPoolExecutor(max_workers=2) as executor:
                train_loader, val_loader = executor.map(load_dataset, ['train', 'val'], [i, i], [device, device])

            print("\nLenght of the datasets:", len(train_loader), len(val_loader))


            for epoch in range(num_epochs_for_each_chunck):
                
                start = datetime.now()
                model.train()
                train_loss = 0
                for data in train_loader:
                    inputs, targets = data
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        inputs, targets = data
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                scheduler.step(val_loss)
                print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    early_stopping_triggered = True
                    break
                print(f"Time to move data to GPU: {datetime.now() - start}")

    print("\n-------------------------------------------")
    print("Training completed")
    print("-------------------------------------------\n") 

    gc.collect()

    test_losses = []
    i = 0

    for i in range(number_of_chucks_testset): #type: ignore

        print(f"\nTest chunck number {i+1} of {number_of_chucks_testset}: ")

        name_test = f"dataset_test{i}.beton"  # Define the path where the dataset will be written
        complete_path_train = os.path.join(FFCV_DIR, name_test)

        test_loader = load_dataset('test', i, device)

        print("\nLenght test dataset: ", len(test_loader))

        gc.collect()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{total_loss:.4f}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved')