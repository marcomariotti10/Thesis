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
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler
from functions_for_NN import *
from constants import *

if __name__ == "__main__":
    
    gc.collect()

    random.seed(SEED)

    # Model creation
    model = Autoencoder()
    model.apply(weights_init)
    criterion = WeightedCustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

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
    num_total_epochs = 2
    num_epochs_for_each_chunck = 2
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

            indices = [i]  # Load only the current chunk

            dataset = LidarDataset(CHUNCKS_DIR, CHUNCKS_DIR, i)
            split_index = math.ceil(len(dataset) * 0.9)
            dataset, val_dataset = random_split(dataset, [split_index, len(dataset) - split_index])
            print("Lenght training and validation set:", len(dataset), len(val_dataset))

            train_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, pin_memory=True)
            print("\nData loaders created")

            '''
            with ThreadPoolExecutor(max_workers=2) as executor:
                complete_grid_maps, complete_grid_maps_BB = executor.map(load_array, [
                    os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{i}.npy'),
                    os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{i}.npy')
                ])

            # Split the data
            split_index = math.ceil(len(complete_grid_maps) * 0.9)
            X_val = complete_grid_maps[split_index:]
            complete_grid_maps = complete_grid_maps[:split_index]
            y_val = complete_grid_maps_BB[split_index:]
            complete_grid_maps_BB = complete_grid_maps_BB[:split_index]

            gc.collect()

            print("\nDivision between train and val: ", complete_grid_maps.shape, X_val.shape, complete_grid_maps_BB.shape, y_val.shape)
            
            # Prepare data loaders

            train_loader = DataLoader(TensorDataset(torch.from_numpy(complete_grid_maps).float(), torch.from_numpy(complete_grid_maps_BB).float()), batch_size=32, shuffle=False, pin_memory=True)
            val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()), batch_size=32, shuffle=False, pin_memory=True)

            print("\nData loaders created")
            del complete_grid_maps, X_val, complete_grid_maps_BB, y_val
            gc.collect()
            '''

            for epoch in range(num_epochs_for_each_chunck):
                
                model.train()
                train_loss = 0
                for data in train_loader:
                    inputs, targets = data
                    inputs, targets = inputs.to(device, non_blocking = True), targets.to(device)
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
                        inputs, targets = inputs.to(device, non_blocking = True), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                scheduler.step(val_loss)
                print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                torch.cuda.empty_cache()

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    early_stopping_triggered = True
                    break
    print("\n-------------------------------------------")
    print("Training completed")
    print("-------------------------------------------\n") 

    gc.collect()

    test_losses = []
    i = 0

    for i in range(number_of_chucks_testset): #type: ignore

        print(f"\nTest chunck number {i+1} of {number_of_chucks_testset}: ")

        # Load the arrays
        complete_grid_maps = np.load(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_test_{i}.npy'))
        complete_grid_maps_BB = np.load(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_test_{i}.npy'))

        print("\nShapes: ",complete_grid_maps.shape, complete_grid_maps_BB.shape)
            
        # Prepare data loaders

        test_loader = DataLoader(TensorDataset(torch.from_numpy(complete_grid_maps).float(), torch.from_numpy(complete_grid_maps_BB).float()), batch_size=32, shuffle=False, pin_memory=True)

        print("\nTest loader created")
        gc.collect()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{total_loss:.4f}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved')


#TODO: understand why the jupither is much faster