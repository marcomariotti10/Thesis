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

def process_lidar_chunk(lidar_directory, position_directory, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, is_training):
    generate_combined_grid_maps(lidar_directory, position_directory, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, is_training) # type: ignore

    # Info about the number of bounding boxes
    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, position_directory)
    print(f"\nSum_chunck: ", sum_ped, sum_bic, sum_car)
    print(f"Average_chunck: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

    return complete_grid_maps, complete_grid_maps_BB

if __name__ == '__main__':

    number_of_chuncks = NUMBER_OF_CHUNCKS
    number_of_chuncks_test = NUMBER_OF_CHUNCKS_TEST

    gc.collect()

    set_start_method("spawn", force=True)

    random.seed(SEED)

    os.makedirs(CHUNCKS_DIR, exist_ok=True)

    # Load scalers
    with open(os.path.join(SCALER_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(SCALER_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    # Generation chuncks training set

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
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chuncks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chuncks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chuncks) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    for i in range(number_of_chuncks): #type: ignore
        
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_numb_BB = []

        print(f"\nChunck number {i+1} of {number_of_chuncks}")

        files_lidar_chunck_1 = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
        files_BB_chunck_1 = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
        
        files_lidar_chunck_2 = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
        files_BB_chunck_2 = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore

        files_lidar_chunck_3 = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
        files_BB_chunck_3 = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(process_lidar_chunk, LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, files_lidar_chunck_1, files_BB_chunck_1, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))
            futures.append(executor.submit(process_lidar_chunk, LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, files_lidar_chunck_2, files_BB_chunck_2, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))
            futures.append(executor.submit(process_lidar_chunk, LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, files_lidar_chunck_3, files_BB_chunck_3, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))

            for future in futures:
                complete_grid_maps, complete_grid_maps_BB = future.result()

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)
        print(f"\ncomplete grid map shape : {complete_grid_maps.shape}")

        # Concatenate the lists in complete_grid_maps_BB along the first dimension
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        # Number of random samples you want to take
        num_samples = int(complete_grid_maps.shape[0] * 0.2)

        # Generate random indices
        random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

        # Select elements at the random indices
        random_complete_grid_maps = complete_grid_maps[random_indices]
        random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]

        print(f"\nRandom complete grid maps shape: {random_complete_grid_maps.shape}")
        print(f"Random complete grid maps BB shape: {random_complete_grid_maps_BB.shape}")
        
        augmented_grid_maps, augmented_grid_maps_BB = apply_augmentation(random_complete_grid_maps, random_complete_grid_maps_BB)

        augmented_grid_maps = np.array(augmented_grid_maps)
        augmented_grid_maps_BB = np.array(augmented_grid_maps_BB)

        print(f"\nAugmented grid maps shape: {augmented_grid_maps.shape}")
        print(f"Augmented grid maps BB shape: {augmented_grid_maps_BB.shape}")

         # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.concatenate((complete_grid_maps, augmented_grid_maps), axis=0)
        print(f"\nNew complete grid map shape : {complete_grid_maps.shape}")
        complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)
        print(f"New complete grid map BB shape : {complete_grid_maps_BB.shape}")

        del augmented_grid_maps, augmented_grid_maps_BB, random_complete_grid_maps, random_complete_grid_maps_BB
        gc.collect()

        indices = np.arange(complete_grid_maps.shape[0])
        np.random.shuffle(indices)
        complete_grid_maps = complete_grid_maps[indices]
        complete_grid_maps_BB = complete_grid_maps_BB[indices]

        complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

        # Save the arrays
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{i}.npy'), complete_grid_maps)
        print(f"complete grid map {i} saved")
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{i}.npy'), complete_grid_maps_BB)
        print(f"complete grid map BB {i} saved")

    # Generation chuncks test set

    print("\nStarting test set generation")

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
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chuncks_test) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chuncks_test) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chuncks_test) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    gc.collect()

    i = 0

    for i in range (number_of_chuncks_test):

        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_numb_BB = []

        files_lidar_chunck_1 = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
        files_BB_chunck_1 = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
        
        files_lidar_chunck_2 = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
        files_BB_chunck_2 = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore
        
        files_lidar_chunck_3 = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
        files_BB_chunck_3 = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(process_lidar_chunk, LIDAR_1_TEST, POSITION_1_TEST, files_lidar_chunck_1, files_BB_chunck_1, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False)) # type: ignore
            futures.append(executor.submit(process_lidar_chunk, LIDAR_2_TEST, POSITION_2_TEST, files_lidar_chunck_2, files_BB_chunck_2, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False)) # type: ignore
            futures.append(executor.submit(process_lidar_chunk, LIDAR_3_TEST, POSITION_3_TEST, files_lidar_chunck_3, files_BB_chunck_3, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False)) # type: ignore
            
            for future in futures:
                complete_grid_maps, complete_grid_maps_BB = future.result()

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)
        print(f"complete grid map shape : {complete_grid_maps.shape}")

        # Concatenate the lists in complete_grid_maps_BB along the first dimension
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        # Normalize the data
        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        # Save the arrays
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_test_{i}.npy'), complete_grid_maps)
        print(f"complete grid map test {i} saved")
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_test_{i}.npy'), complete_grid_maps_BB)
        print(f"complete grid map BB test {i} saved")
    