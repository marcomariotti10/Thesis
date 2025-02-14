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
import math
from sklearn.preprocessing import MinMaxScaler
from functions_for_NN import *
from constants import *

if __name__ == "__main__":
    
    gc.collect()

    set_start_method("spawn", force=True)

    model = Autoencoder()
    model.apply(weights_init)
    criterion = WeightedCustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

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

    early_stopping_triggered = False
    number_of_chucks= 1
    num_total_epochs = 1
    num_epochs_for_each_chunck = 1

    files_lidar_1 = sorted([f for f in os.listdir(LIDAR_1_GRID_DIRECTORY)]) #type: ignore
    file_BB_1 = sorted([f for f in os.listdir(POSITION_LIDAR_1_GRID_NO_BB)]) #type: ignore

    files_lidar_2 = sorted([f for f in os.listdir(LIDAR_2_GRID_DIRECTORY)]) #type: ignore
    file_BB_2 = sorted([f for f in os.listdir(POSITION_LIDAR_2_GRID_NO_BB)]) #type: ignore

    files_lidar_3 = sorted([f for f in os.listdir(LIDAR_3_GRID_DIRECTORY)]) #type: ignore
    file_BB_3 = sorted([f for f in os.listdir(POSITION_LIDAR_3_GRID_NO_BB)]) #type: ignore

    total_num_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"Total number of files: {total_num_files1, total_num_of_files2, total_num_of_files3}")

    file_for_chunck1 = math.ceil(total_num_files1 / number_of_chucks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chucks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chucks) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    for j in range(num_total_epochs):
        if early_stopping_triggered:
            break
        for i in range(number_of_chucks): #type: ignore
            if early_stopping_triggered:
                break
            complete_grid_maps = []
            complete_grid_maps_BB = []
            complete_num_BB = []

            files_lidar = take_split_files(files_lidar_1, file_for_chunck1, i)
            files_BB = take_split_files(file_BB_1, file_for_chunck1, i)

            generate_combined_grid_maps(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, files_lidar, files_BB, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
            
            files_lidar = take_split_files(files_lidar_2, file_for_chunck2, i)
            files_BB = take_split_files(file_BB_2, file_for_chunck2, i)

            generate_combined_grid_maps(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, files_lidar, files_BB, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
            
            files_lidar = take_split_files(files_lidar_3, file_for_chunck3, i)
            files_BB = take_split_files(file_BB_3, file_for_chunck3, i)

            generate_combined_grid_maps(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, files_lidar, files_BB, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
            
            del files_lidar, files_BB

            print("Chunck number:", i+1)

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

            # Visualize data with and without bounding boxes
            total_num_greater_than_1 = 0
            total_num_equal_to_0 = 0
            for i in range(complete_num_BB.shape[0]):
                if (complete_num_BB[i] >= 1):
                    total_num_greater_than_1 += 1
                else:
                    total_num_equal_to_0 += 1

            print(total_num_greater_than_1, total_num_equal_to_0)

            # Split the data
            X_train, X_val, y_train, y_val, num_BB_train, num_BB_val = split_data(complete_grid_maps, complete_grid_maps_BB, complete_num_BB, TEST_SIZE) # type: ignore

            del complete_grid_maps, complete_grid_maps_BB, complete_num_BB

            print(X_train.shape, X_val.shape, y_train.shape, y_val.shape, num_BB_train.shape, num_BB_val.shape)

            sum = visualize_proportion(num_BB_train)
            print(f"Sum_train: ", sum)
            print(f"Average_sum_train: ", sum/len(num_BB_train))

            sum = visualize_proportion(num_BB_val)
            print(f"Sum_train: ", sum)
            print(f"Average_sum_train: ", sum/len(num_BB_val))

            # Initialize the scaler
            scaler = MinMaxScaler()

            # Normalize the data
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

            y_train = scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
            y_val = scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)

            print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)

            print("New input shape (train, val):", X_train.shape, X_val.shape)

            y_train = np.expand_dims(y_train, axis=1)
            y_val = np.expand_dims(y_val, axis=1)

            print("New labels shape (train, val):", y_train.shape, y_val.shape)
                
            # Prepare data loaders
            train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.
            val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
            
            del X_train, X_val, y_train, y_val

            print("Len dataset (train, val):",len(train_dataset), len(val_dataset))

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            del train_dataset, val_dataset

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
    
    print("Training completed")

    complete_grid_maps = []
    complete_grid_maps_BB = []
    complete_num_BB = []

    files_lidar = sorted([f for f in os.listdir(LIDAR_1_TEST)]) #type: ignore
    files_BB = sorted([f for f in os.listdir(POSITION_1_TEST)]) #type: ignore

    generate_combined_grid_maps(LIDAR_1_TEST, POSITION_1_TEST, files_lidar, files_BB, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
    
    files_lidar = sorted([f for f in os.listdir(LIDAR_2_TEST)]) #type: ignore
    files_BB = sorted([f for f in os.listdir(POSITION_2_TEST)]) #type: ignore

    generate_combined_grid_maps(LIDAR_2_TEST, POSITION_2_TEST, files_lidar, files_BB, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
    
    files_lidar = sorted([f for f in os.listdir(LIDAR_3_TEST)]) #type: ignore
    files_BB = sorted([f for f in os.listdir(POSITION_3_TEST)]) #type: ignore

    generate_combined_grid_maps(LIDAR_3_TEST, POSITION_3_TEST, files_lidar, files_BB, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
    
    del files_lidar, files_BB

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

    # Visualize data with and without bounding boxes
    total_num_greater_than_1 = 0
    total_num_equal_to_0 = 0
    for i in range(complete_num_BB.shape[0]):
        if (complete_num_BB[i] >= 1):
            total_num_greater_than_1 += 1
        else:
            total_num_equal_to_0 += 1

    print(total_num_greater_than_1, total_num_equal_to_0)

    # Split the data
    sum = visualize_proportion(complete_num_BB)
    print(f"Sum_test: ", sum)
    print(f"Average_sum_test: ", sum/len(complete_num_BB))

    del complete_num_BB

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Normalize the data
    complete_grid_maps = scaler.fit_transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)
    
    complete_grid_maps_BB = scaler.transform(complete_grid_maps_BB.reshape(-1, complete_grid_maps_BB.shape[-1])).reshape(complete_grid_maps_BB.shape)

    print(complete_grid_maps.shape, complete_grid_maps_BB.shape)

    complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
    complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

    print("New shape test:", complete_grid_maps.shape, complete_grid_maps_BB.shape)
        
    # Prepare data loaders
    test_dataset = TensorDataset(torch.from_numpy(complete_grid_maps).float(), torch.from_numpy(complete_grid_maps_BB).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.
    
    del complete_grid_maps, complete_grid_maps_BB

    print("Len dataset test:", len(test_dataset))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    del test_dataset

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
