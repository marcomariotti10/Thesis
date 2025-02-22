import torch
from ffcv.fields import NDArrayField, FloatField
from ffcv.writer import DatasetWriter
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

class Dataset:
    def __init__(self, files_lidar_1, files_BB_1, files_lidar_2, files_BB_2, files_lidar_3, files_BB_3, scaler_X, num_samples_ratio=0.2):
        self.files_lidar_1 = files_lidar_1
        self.files_BB_1 = files_BB_1
        self.files_lidar_2 = files_lidar_2
        self.files_BB_2 = files_BB_2
        self.files_lidar_3 = files_lidar_3
        self.files_BB_3 = files_BB_3
        self.scaler_X = scaler_X
        self.num_samples_ratio = num_samples_ratio
        self.complete_grid_maps, self.complete_grid_maps_BB = self._generate_data()

    def _generate_data(self):

        if HOME[0] == 'C':
            lidar_1_grid_directory = '/mnt/c' + LIDAR_1_GRID_DIRECTORY[2:]
            position_lidar_1_grid_no_bb = '/mnt/c' + POSITION_LIDAR_1_GRID_NO_BB[2:]
            lidar_2_grid_directory = '/mnt/c' + LIDAR_2_GRID_DIRECTORY[2:]
            position_lidar_2_grid_no_bb = '/mnt/c' + POSITION_LIDAR_2_GRID_NO_BB[2:]
            lidar_3_grid_directory = '/mnt/c' + LIDAR_3_GRID_DIRECTORY[2:]
            position_lidar_3_grid_no_bb = '/mnt/c' + POSITION_LIDAR_3_GRID_NO_BB[2:]
        else:
            lidar_1_grid_directory = LIDAR_1_GRID_DIRECTORY
            position_lidar_1_grid_no_bb = POSITION_LIDAR_1_GRID_NO_BB
            lidar_2_grid_directory = LIDAR_2_GRID_DIRECTORY
            position_lidar_2_grid_no_bb = POSITION_LIDAR_2_GRID_NO_BB
            lidar_3_grid_directory = LIDAR_3_GRID_DIRECTORY
            position_lidar_3_grid_no_bb = POSITION_LIDAR_3_GRID_NO_BB
        
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_numb_BB = []

        # Number of files of each chunck for each lidar
        file_for_chunck1 = math.ceil(len(self.files_lidar_1) / NUMBER_OF_CHUNCKS) #type: ignore
        file_for_chunck2 = math.ceil(len(self.files_lidar_2) / NUMBER_OF_CHUNCKS) #type: ignore
        file_for_chunck3 = math.ceil(len(self.files_lidar_3) / NUMBER_OF_CHUNCKS) #type: ignore

        print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

        files_lidar_chunck_1 = self.files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(self.files_lidar_1) ) ] #type: ignore
        files_BB_chunck_1 = self.files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(self.files_BB_1) ) ] #type: ignore
        
        files_lidar_chunck_2 = self.files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(self.files_lidar_2) ) ] #type: ignore
        files_BB_chunck_2 = self.files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(self.files_BB_2) ) ] #type: ignore

        files_lidar_chunck_3 = self.files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(self.files_lidar_3) ) ] #type: ignore
        files_BB_chunck_3 = self.files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(self.files_BB_3) ) ] #type: ignore

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(process_lidar_chunk, lidar_1_grid_directory, position_lidar_1_grid_no_bb, files_lidar_chunck_1, files_BB_chunck_1, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))
            futures.append(executor.submit(process_lidar_chunk, lidar_2_grid_directory, position_lidar_2_grid_no_bb, files_lidar_chunck_2, files_BB_chunck_2, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))
            futures.append(executor.submit(process_lidar_chunk, lidar_3_grid_directory, position_lidar_3_grid_no_bb, files_lidar_chunck_3, files_BB_chunck_3, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))

            for future in futures:
                complete_grid_maps, complete_grid_maps_BB = future.result()

        complete_grid_maps = np.array(complete_grid_maps)
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)

        complete_grid_maps = self.scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        num_samples = int(complete_grid_maps.shape[0] * self.num_samples_ratio)
        random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

        random_complete_grid_maps = complete_grid_maps[random_indices]
        random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]

        augmented_grid_maps, augmented_grid_maps_BB = apply_augmentation(random_complete_grid_maps, random_complete_grid_maps_BB)

        augmented_grid_maps = np.array(augmented_grid_maps)
        augmented_grid_maps_BB = np.array(augmented_grid_maps_BB)

        complete_grid_maps = np.concatenate((complete_grid_maps, augmented_grid_maps), axis=0)
        complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)

        indices = np.arange(complete_grid_maps.shape[0])
        np.random.shuffle(indices)
        complete_grid_maps = complete_grid_maps[indices]
        complete_grid_maps_BB = complete_grid_maps_BB[indices]

        complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

        del augmented_grid_maps, augmented_grid_maps_BB, random_complete_grid_maps, random_complete_grid_maps_BB, complete_numb_BB, files_lidar_chunck_1, files_BB_chunck_1, files_lidar_chunck_2, files_BB_chunck_2, files_lidar_chunck_3, files_BB_chunck_3
        gc.collect()
        
        return complete_grid_maps, complete_grid_maps_BB

    def __getitem__(self, idx):
        return (self.complete_grid_maps[idx].astype('float32'), self.complete_grid_maps_BB[idx].astype('float32'))

    def __len__(self):
        return len(self.complete_grid_maps)

if __name__ == '__main__':

    number_of_chuncks = NUMBER_OF_CHUNCKS
    number_of_chuncks_test = NUMBER_OF_CHUNCKS_TEST

    gc.collect()

    set_start_method("spawn", force=True)

    random.seed(SEED)          

    if HOME[0] == 'C':
        scaler_dir = '/mnt/c' + SCALER_DIR[2:]
        write_path = '/mnt/c' + FFCV_DIR[2:]
        lidar_1_grid_directory = '/mnt/c' + LIDAR_1_GRID_DIRECTORY[2:]
        position_lidar_1_grid_no_bb = '/mnt/c' + POSITION_LIDAR_1_GRID_NO_BB[2:]
        lidar_2_grid_directory = '/mnt/c' + LIDAR_2_GRID_DIRECTORY[2:]
        position_lidar_2_grid_no_bb = '/mnt/c' + POSITION_LIDAR_2_GRID_NO_BB[2:]
        lidar_3_grid_directory = '/mnt/c' + LIDAR_3_GRID_DIRECTORY[2:]
        position_lidar_3_grid_no_bb = '/mnt/c' + POSITION_LIDAR_3_GRID_NO_BB[2:]
    else:
        write_path = FFCV_DIR
        scaler_dir = SCALER_DIR
        lidar_1_grid_directory = LIDAR_1_GRID_DIRECTORY
        position_lidar_1_grid_no_bb = POSITION_LIDAR_1_GRID_NO_BB
        lidar_2_grid_directory = LIDAR_2_GRID_DIRECTORY
        position_lidar_2_grid_no_bb = POSITION_LIDAR_2_GRID_NO_BB
        lidar_3_grid_directory = LIDAR_3_GRID_DIRECTORY
        position_lidar_3_grid_no_bb = POSITION_LIDAR_3_GRID_NO_BB

    os.makedirs(write_path, exist_ok=True)

    # Load scalers
    with open(os.path.join(scaler_dir, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)

    # Generation chuncks training set

    # Shuffle files_lidar_1 and files_BB_1 in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(lidar_1_grid_directory)]), sorted([f for f in os.listdir(position_lidar_1_grid_no_bb)])))
    random.shuffle(combined_files)
    files_lidar_1, files_BB_1 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_1 = list(files_lidar_1)
    files_BB_1 = list(files_BB_1)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, position_lidar_1_grid_no_bb)
    print(f"\nSum_complete_lidar1: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))

    combined_files = list(zip(sorted([f for f in os.listdir(lidar_2_grid_directory)]), sorted([f for f in os.listdir(position_lidar_2_grid_no_bb)])))
    random.shuffle(combined_files)
    files_lidar_2, files_BB_2 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_2 = list(files_lidar_2)
    files_BB_2 = list(files_BB_2)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_2, position_lidar_2_grid_no_bb)
    print(f"\nSum_complete_lidar2: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar2: ", sum_ped/len(files_BB_2), sum_bic/len(files_BB_2), sum_car/len(files_BB_2))

    combined_files = list(zip(sorted([f for f in os.listdir(lidar_3_grid_directory)]), sorted([f for f in os.listdir(position_lidar_3_grid_no_bb)])))
    random.shuffle(combined_files)
    files_lidar_3, files_BB_3 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_3 = list(files_lidar_3)
    files_BB_3 = list(files_BB_3)

    del combined_files
    gc.collect()

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, position_lidar_3_grid_no_bb)
    print(f"\nSum_complete_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")
        

    for i in range(number_of_chuncks): #type: ignore

        print(f"\nChunck number {i+1} of {number_of_chuncks}")

        dataset = Dataset(files_lidar_1, files_BB_1, files_lidar_2, files_BB_2, files_lidar_3, files_BB_3, scaler_X, num_samples_ratio=0.2)

        name = f"chunck_{i}.beton"  # Define the path where the dataset will be writtenù

        complete_path = os.path.join(write_path, name)

        shape = (X_RANGE, Y_RANGE)  # Shape of each sample

        writer = DatasetWriter(complete_path, {
            'covariate': NDArrayField(shape=shape, dtype=np.dtype('float32')),  # Adjust shape
            'label': NDArrayField(shape=shape, dtype=np.dtype('float32')),
        }, num_workers=3)

        writer.from_indexed_dataset(dataset)

        del dataset

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