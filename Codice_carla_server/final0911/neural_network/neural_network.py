import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv
import os
import numpy as np
import sys
from sklearn.model_selection import train_test_split
import importlib
from torchsummary import summary
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
import gc
from multiprocessing import Pool, set_start_method
from matplotlib.path import Path
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MinMaxScaler
from functions_for_NN import *
from constants import *

if __name__ == "__main__":
    
    gc.collect()

    set_start_method("spawn", force=True)

    complete_grid_maps = []
    complete_grid_maps_BB = []
    complete_num_BB = []

    # Load sensor1
    grid_map_files1 = sorted([f for f in os.listdir(LIDAR_1_GRID_DIRECTORY)]) # type: ignore
    grid_map_BB_files1 = sorted([f for f in os.listdir(POSITION_LIDAR_1_GRID_NO_BB)]) # type: ignore

    generate_combined_grid_maps(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, grid_map_files1, grid_map_BB_files1, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

    del grid_map_files1, grid_map_BB_files1

    # Load sensor2
    grid_map_files2 = sorted([f for f in os.listdir(LIDAR_2_GRID_DIRECTORY)]) # type: ignore
    grid_map_BB_files2 = sorted([f for f in os.listdir(POSITION_LIDAR_2_GRID_NO_BB)]) # type: ignore

    generate_combined_grid_maps(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, grid_map_files2, grid_map_BB_files2, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

    del grid_map_files2, grid_map_BB_files2

    # Load sensor3
    grid_map_files3 = sorted([f for f in os.listdir(LIDAR_3_GRID_DIRECTORY)]) # type: ignore
    grid_map_BB_files3 = sorted([f for f in os.listdir(POSITION_LIDAR_3_GRID_NO_BB)]) # type: ignore

    generate_combined_grid_maps(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, grid_map_files3, grid_map_BB_files3, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

    del grid_map_files3, grid_map_BB_files3

    #print(f"Number of grid maps for each sensor: {len(complete_grid_maps[0])}, {len(complete_grid_maps[1])}, {len(complete_grid_maps[2])}")
    #print(f"Number of grid maps BB for each sensor: {len(complete_grid_maps_BB[0])}, {len(complete_grid_maps_BB[1])}, {len(complete_grid_maps_BB[2])}")
    #print(f"Number of bounding boxes for each sensor: {len(complete_num_BB[0])}, {len(complete_num_BB[1])}, {len(complete_num_BB[2])}")

    # Concatenate the lists in complete_grid_maps along the first dimension
    complete_grid_maps = np.array(complete_grid_maps)
    print(f"complete grid map shape : {complete_grid_maps.shape}")

    # Concatenate the lists in complete_grid_maps_BB along the first dimension
    complete_grid_maps_BB = np.array(complete_grid_maps_BB)
    print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

    complete_num_BB = np.array(complete_num_BB)
    print(f"complete number of bounding boxes shape : {complete_num_BB.shape}")

    complete_num_BB = np.expand_dims(complete_num_BB, axis=1)
    print(f"expanded number of bounding boxes shape : {complete_num_BB.shape}")

    # ELiminate all the data points with 0 bounding boxes
    total_num_greater_than_1 = 0
    total_num_equal_to_0 = 0
    for i in range(complete_num_BB.shape[0]):
        if (complete_num_BB[i] >= 1):
            total_num_greater_than_1 += 1
        else:
            total_num_equal_to_0 += 1

    print(total_num_greater_than_1, total_num_equal_to_0)

    # Split the data
    X_train_val, X_test, y_train_val, y_test, num_BB_train_val, num_BB_test = split_data(complete_grid_maps, complete_grid_maps_BB, complete_num_BB, TEST_SIZE) # type: ignore

    del complete_grid_maps, complete_grid_maps_BB, complete_num_BB

    X_train, X_val, y_train, y_val, num_BB_train, num_BB_val = split_data(X_train_val, y_train_val, num_BB_train_val, len(X_test)) # Esure that val and test set have the same lenght

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape, num_BB_train.shape, num_BB_val.shape, num_BB_test.shape)
    
    sum_train = 0
    sum_val = 0
    sum_test = 0

    for i in range(len(num_BB_train)):
        sum_train += num_BB_train[i]
    print(f"Sum_train: ", sum_train)
    print(f"Average_sum_train: ",sum_train/len(num_BB_train))

    for i in range(len(num_BB_val)):
        sum_val += num_BB_val[i]
    print(f"Sum_val: ",sum_val)
    print(f"Average_sum_val: ",sum_val/len(num_BB_val))

    for i in range(len(num_BB_test)):
        sum_test += num_BB_test[i]
    print(f"Sum_test: ",sum_test)
    print(f"Average_sum_test: ",sum_test/len(num_BB_test))

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Normalize the data
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train = scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_val = scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
    y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    print("New input shape (train, val, test):", X_train.shape, X_val.shape, X_test.shape)

    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    print("New labels shape (train, val, test):", y_train.shape, y_val.shape, y_test.shape)
        
    # Prepare data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    del X_train, X_val, X_test, y_train, y_val, y_test

    print("Len dataset (train, val, test):",len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    del train_dataset, val_dataset, test_dataset

    model = Autoencoder()
    model.apply(weights_init)
    criterion = WeightedCustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # model.parameters() passes the parameters of the model to the optimizer so that it can update them during training.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    summary(model, (1, 400, 400))
    
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

    num_epochs = 50
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)  # Average loss over all batches

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)  # Average loss over all batches

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

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
    # Clear GPU cache
    torch.cuda.empty_cache()


    # Make predictions
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs= inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions).cpu().numpy()
    print("Predictions Shape:", predictions.shape)
    # Clear GPU cache
    torch.cuda.empty_cache()
