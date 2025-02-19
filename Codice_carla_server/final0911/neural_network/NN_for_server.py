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

            complete_grid_maps = []
            complete_grid_maps_BB = []

            # Load the arrays
            complete_grid_maps = np.load(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{i}.npy'))
            complete_grid_maps_BB = np.load(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{i}.npy'))

            # Number of random samples you want to take
            num_samples = int(complete_grid_maps.shape[0] * 0.1)

            # Generate random indices
            random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

            # Select elements at the random indices
            random_complete_grid_maps = complete_grid_maps[random_indices]
            random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]

            print(f"\nRandom complete grid maps shape: {random_complete_grid_maps.shape}")
            print(f"Random complete grid maps BB shape: {random_complete_grid_maps_BB.shape}")
            
            augmented_grid_maps, augmented_grid_maps_BB = apply_augmentation(random_complete_grid_maps, random_complete_grid_maps_BB, j)

            augmented_grid_maps = np.array(augmented_grid_maps)
            augmented_grid_maps_BB = np.array(augmented_grid_maps_BB)

            print(f"\nAugmented grid maps shape: {augmented_grid_maps.shape}")
            print(f"Augmented grid maps BB shape: {augmented_grid_maps_BB.shape}")

            '''
            for i in range(num_samples):
                print("point BB original:", random_complete_grid_maps_BB[i][0][0], random_complete_grid_maps_BB[i][399][0], random_complete_grid_maps_BB[i][0][399], random_complete_grid_maps_BB[i][399][399])
                print("point BB augmented:", augmented_grid_maps_BB[i][0][0], augmented_grid_maps_BB[i][399][0], augmented_grid_maps_BB[i][0][399], augmented_grid_maps_BB[i][399][399])

                print("point original:", random_complete_grid_maps[i][0][0], random_complete_grid_maps[i][399][0], random_complete_grid_maps[i][0][399], random_complete_grid_maps[i][399][399])
                print("point augmented:", augmented_grid_maps[i][0][0], augmented_grid_maps[i][399][0], augmented_grid_maps[i][0][399], augmented_grid_maps[i][399][399])
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
                ax[0, 0].imshow(random_complete_grid_maps[i], cmap='gray')
                ax[0, 0].set_title('Original Grid Map')
                
                ax[0, 1].imshow(random_complete_grid_maps_BB[i], cmap='gray')
                ax[0, 1].set_title('Original Grid Map BB')
                
                ax[1, 0].imshow(augmented_grid_maps[i], cmap='gray')
                ax[1, 0].set_title(f'Augmented Grid Map')
                
                ax[1, 1].imshow(augmented_grid_maps_BB[i], cmap='gray')
                ax[1, 1].set_title(f'Augmented Grid Map BB')
        
                plt.show()
            '''

            # Concatenate the lists in complete_grid_maps along the first dimension
            complete_grid_maps = np.concatenate((complete_grid_maps, augmented_grid_maps), axis=0)
            print(f"\nNew complete grid map shape : {complete_grid_maps.shape}")
            complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)
            print(f"New complete grid map BB shape : {complete_grid_maps_BB.shape}")

            del augmented_grid_maps, augmented_grid_maps_BB, random_complete_grid_maps, random_complete_grid_maps_BB
            gc.collect()

            # Shuffle the data
            combined_files = list(zip(complete_grid_maps, complete_grid_maps_BB))
            random.shuffle(combined_files)
            complete_grid_maps, complete_grid_maps_BB = zip(*combined_files)
            # Convert back to lists if needed
            complete_grid_maps = np.array(complete_grid_maps)
            complete_grid_maps_BB = np.array(complete_grid_maps_BB)

            del combined_files
            gc.collect()

            # Split the data
            X_train = complete_grid_maps[0:math.ceil((len(complete_grid_maps)*0.9))]
            X_val = complete_grid_maps[math.ceil((len(complete_grid_maps)*0.9)):len(complete_grid_maps)] 
            y_train = complete_grid_maps_BB[0:math.ceil((len(complete_grid_maps)*0.9))]
            y_val = complete_grid_maps_BB[math.ceil((len(complete_grid_maps)*0.9)):len(complete_grid_maps)]

            del complete_grid_maps, complete_grid_maps_BB
            gc.collect()

            print("\nDivision between train and val: ", X_train.shape, X_val.shape, y_train.shape, y_val.shape)

            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)

            print("\nNew input shape (train, val):", X_train.shape, X_val.shape)

            y_train = np.expand_dims(y_train, axis=1)
            y_val = np.expand_dims(y_val, axis=1)

            print("New labels shape (train, val):", y_train.shape, y_val.shape)
                
            # Prepare data loaders
            train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.
            val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
            
            del X_train, X_val, y_train, y_val
            gc.collect()

            print("\nLenght dataset (train, val):",len(train_dataset), len(val_dataset))

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            del train_dataset, val_dataset
            gc.collect()

            for epoch in range(num_epochs_for_each_chunck):
                
                model.train()
                train_loss = 0
                for data in train_loader:
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)
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
                        inputs, targets = inputs.to(device), targets.to(device)
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

        complete_grid_maps = []
        complete_grid_maps_BB = []

        # Load the arrays
        complete_grid_maps = np.load(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_test_{i}.npy'))
        complete_grid_maps_BB = np.load(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_test_{i}.npy'))

        print("\nShapes: ",complete_grid_maps.shape, complete_grid_maps_BB.shape)

        complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

        print("New shape test:", complete_grid_maps.shape, complete_grid_maps_BB.shape)
            
        # Prepare data loaders
        test_dataset = TensorDataset(torch.from_numpy(complete_grid_maps).float(), torch.from_numpy(complete_grid_maps_BB).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.

        #del complete_grid_maps, complete_grid_maps_BB

        print("\nLenght dataset test:", len(test_dataset))

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        del test_dataset
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

    total_loss = 0
    for k in range(len(test_losses)):
        total_loss += test_losses[k]
    total_loss /= len(test_losses)
    print("\nTotal test losses: ", total_loss)

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{total_loss:.4f}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved')


#TODO: understand why the jupither is much faster