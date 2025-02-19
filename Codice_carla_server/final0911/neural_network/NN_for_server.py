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

    set_start_method("spawn", force=True)

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
    number_of_chucks= 1
    num_total_epochs = 1
    num_epochs_for_each_chunck = 50
    number_of_chucks_testset = 1

    # Load scalers
    with open(os.path.join(SCALER_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(SCALER_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    # Shuffle files_lidar_1 and files_BB_1 in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_1_GRID_DIRECTORY)]), sorted([f for f in os.listdir(POSITION_LIDAR_1_GRID_NO_BB)])))
    random.shuffle(combined_files)
    files_lidar_1, files_BB_1 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_1 = list(files_lidar_1)
    files_BB_1 = list(files_BB_1)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, POSITION_LIDAR_1_GRID_NO_BB)
    print(f"\nSum_complete_lidar1: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_2_GRID_DIRECTORY)]), sorted([f for f in os.listdir(POSITION_LIDAR_2_GRID_NO_BB)])))
    random.shuffle(combined_files)
    files_lidar_2, files_BB_2 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_2 = list(files_lidar_2)
    files_BB_2 = list(files_BB_2)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_2, POSITION_LIDAR_2_GRID_NO_BB)
    print(f"\nSum_complete_lidar2: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar2: ", sum_ped/len(files_BB_2), sum_bic/len(files_BB_2), sum_car/len(files_BB_2))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_3_GRID_DIRECTORY)]), sorted([f for f in os.listdir(POSITION_LIDAR_3_GRID_NO_BB)])))
    random.shuffle(combined_files)
    files_lidar_3, files_BB_3 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_3 = list(files_lidar_3)
    files_BB_3 = list(files_BB_3)

    del combined_files
    gc.collect()

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, POSITION_LIDAR_3_GRID_NO_BB)
    print(f"\nSum_complete_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    # Number of files of each chunck for each lidar
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chucks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chucks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chucks) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    for j in range(num_total_epochs):
        
        if early_stopping_triggered:
            break

        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")
        random.seed(SEED + j)

        for i in range(number_of_chucks): #type: ignore
            
            if early_stopping_triggered:
                break
            
            complete_grid_maps = []
            complete_grid_maps_BB = []
            complete_numb_BB = []

            print(f"\nChunck number {i+1} of {number_of_chucks}: ")

            files_lidar_chunck = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
            files_BB_chunck = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
            start = datetime.now()
            generate_combined_grid_maps(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, True) # type: ignore
            print(f"Time to generate combined grid maps: {datetime.now() - start}")

            # Info for lidar 1 about the number of bounding boxes
            sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_LIDAR_1_GRID_NO_BB)
            print(f"\nSum_chunck_lidar1: ", sum_ped, sum_bic, sum_car)
            print(f"Average_chunck_lidar1: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

            files_lidar_chunck = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
            files_BB_chunck = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore
            generate_combined_grid_maps(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, True) # type: ignore
            
            # Info for lidar 2 about the number of bounding boxes
            sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_LIDAR_2_GRID_NO_BB)
            print(f"\nSum_chunck_lidar2: ", sum_ped, sum_bic, sum_car)
            print(f"Average_chunck_lidar2: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

            files_lidar_chunck = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
            files_BB_chunck = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore
            generate_combined_grid_maps(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, True) # type: ignore

            # Info for lidar 3 about the number of bounding boxes
            sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_LIDAR_3_GRID_NO_BB)
            print(f"\nSum_chunck_lidar3: ", sum_ped, sum_bic, sum_car)
            print(f"Average_chunck_lidar3: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

            # Concatenate the lists in complete_grid_maps along the first dimension
            complete_grid_maps = np.array(complete_grid_maps)
            print(f"\ncomplete grid map shape : {complete_grid_maps.shape}")

            # Concatenate the lists in complete_grid_maps_BB along the first dimension
            complete_grid_maps_BB = np.array(complete_grid_maps_BB)
            print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

            complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)
            #complete_grid_maps_BB = scaler_y.transform(complete_grid_maps_BB.reshape(-1, complete_grid_maps_BB.shape[-1])).reshape(complete_grid_maps_BB.shape)

            # Number of random samples you want to take
            num_samples = int(complete_grid_maps.shape[0] * 0.1)

            # Generate random indices
            random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

            # Select elements at the random indices
            random_complete_grid_maps = complete_grid_maps[random_indices]
            random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]
            random_complete_numb_BB = [complete_numb_BB[i] for i in random_indices]

            print(f"\nRandom complete grid maps shape: {random_complete_grid_maps.shape}")
            print(f"Random complete grid maps BB shape: {random_complete_grid_maps_BB.shape}")
            print(f"Random complete number of bounding boxes lenght : {len(random_complete_numb_BB)}")
            
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
            print(f"\nNeq complete grid map shape : {complete_grid_maps.shape}")
            complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)
            print(f"new complete grid map BB shape : {complete_grid_maps_BB.shape}")
            complete_numb_BB.extend(random_complete_numb_BB)
            print(f"new complete number of bounding boxes lenght : {len(complete_numb_BB)}")

            del augmented_grid_maps, augmented_grid_maps_BB, random_complete_grid_maps, random_complete_grid_maps_BB, random_complete_numb_BB
            gc.collect()

            # Shuffle the data
            combined_files = list(zip(complete_grid_maps, complete_grid_maps_BB, complete_numb_BB))
            random.shuffle(combined_files)
            complete_grid_maps, complete_grid_maps_BB, complete_numb_BB = zip(*combined_files)
            # Convert back to lists if needed
            complete_grid_maps = np.array(complete_grid_maps)
            complete_grid_maps_BB = np.array(complete_grid_maps_BB)
            complete_numb_BB = np.array(complete_numb_BB)

            del combined_files
            gc.collect()

            # Split the data
            X_train = complete_grid_maps[0:math.ceil((len(complete_grid_maps)*0.9))]
            X_val = complete_grid_maps[math.ceil((len(complete_grid_maps)*0.9)):len(complete_grid_maps)] 
            y_train = complete_grid_maps_BB[0:math.ceil((len(complete_grid_maps)*0.9))]
            y_val = complete_grid_maps_BB[math.ceil((len(complete_grid_maps)*0.9)):len(complete_grid_maps)]

            complete_numb_BB_train = complete_numb_BB[0:math.ceil((len(complete_grid_maps)*0.9))]
            complete_numb_BB_val = complete_numb_BB[math.ceil((len(complete_grid_maps)*0.9)):len(complete_grid_maps)]

            sum_ped = np.sum(complete_numb_BB_train[:, 0])
            sum_bic = np.sum(complete_numb_BB_train[:, 1])
            sum_car = np.sum(complete_numb_BB_train[:, 2])
            print(f"\nSum_train: ", sum_ped, sum_bic, sum_car)
            print(f"Average_train: ", sum_ped/len(complete_numb_BB_train), sum_bic/len(complete_numb_BB_train), sum_car/len(complete_numb_BB_train))

            sum_ped = np.sum(complete_numb_BB_val[:, 0])
            sum_bic = np.sum(complete_numb_BB_val[:, 1])
            sum_car = np.sum(complete_numb_BB_val[:, 2])
            print(f"\nSum_val: ", sum_ped, sum_bic, sum_car)
            print(f"Average_val: ", sum_ped/len(complete_numb_BB_val), sum_bic/len(complete_numb_BB_val), sum_car/len(complete_numb_BB_val))


            del complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, complete_numb_BB_train, complete_numb_BB_val
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

    # Shuffle files_lidar_1 and files_BB_1 in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_1_TEST)]), sorted([f for f in os.listdir(POSITION_1_TEST)])))
    random.shuffle(combined_files)
    files_lidar_1, files_BB_1 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_1 = list(files_lidar_1)
    files_BB_1 = list(files_BB_1)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, POSITION_1_TEST)
    print(f"\nSum_complete_test_lidar1: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_test_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_2_TEST)]), sorted([f for f in os.listdir(POSITION_2_TEST)])))
    random.shuffle(combined_files)
    files_lidar_2, files_BB_2 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_2 = list(files_lidar_2)
    files_BB_2 = list(files_BB_2)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_2, POSITION_2_TEST)
    print(f"\nSum_complete_test_lidar2: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_test_lidar2: ", sum_ped/len(files_BB_2), sum_bic/len(files_BB_2), sum_car/len(files_BB_2))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_3_TEST)]), sorted([f for f in os.listdir(POSITION_3_TEST)])))
    random.shuffle(combined_files)
    files_lidar_3, files_BB_3 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_3 = list(files_lidar_3)
    files_BB_3 = list(files_BB_3)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, POSITION_3_TEST)
    print(f"\nSum_complete_test_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_test_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    # Number of files of each chunck for each lidar
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chucks_testset) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chucks_testset) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chucks_testset) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    gc.collect()

    test_loss = 0
    predictions = []

    i = 0

    for i in range(number_of_chucks_testset): #type: ignore
        
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_numb_BB = []

        print(f"\nTest chunck number {i+1} of {number_of_chucks_testset}: ")

        files_lidar_chunck = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
        files_BB_chunck = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_1_TEST, POSITION_1_TEST, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_1_TEST)
        print(f"\nSum_chunck_test_lidar1: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_test_lidar1: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))
        
        files_lidar_chunck = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
        files_BB_chunck = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_2_TEST, POSITION_2_TEST, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False) # type: ignore
        
        # Info for lidar 2 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_2_TEST)
        print(f"\nSum_chunck_test_lidar2: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_test_lidar2: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        files_lidar_chunck = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
        files_BB_chunck = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_3_TEST, POSITION_3_TEST, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_3_TEST)
        print(f"\nSum_chunck_test_lidar3: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_test_lidar3: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)
        print(f"complete grid map shape : {complete_grid_maps.shape}")

        # Concatenate the lists in complete_grid_maps_BB along the first dimension
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        #complete_num_BB = np.expand_dims(complete_num_BB, axis=1)
        #print(f"expanded number of bounding boxes shape : {complete_num_BB.shape}")

        gc.collect()

        # Normalize the data
        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        #complete_grid_maps_BB = scaler_y.transform(complete_grid_maps_BB.reshape(-1, complete_grid_maps_BB.shape[-1])).reshape(complete_grid_maps_BB.shape)

        print("\nShape after transform: ",complete_grid_maps.shape, complete_grid_maps_BB.shape)

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

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{test_loss:.4f}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved')


#TODO: understand why the jupither is much faster