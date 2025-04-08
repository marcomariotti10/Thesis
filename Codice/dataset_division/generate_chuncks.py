import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
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
import matplotlib.pyplot as plt
import cv2
import pickle
import random
import math
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler


def process_lidar_chunk(lidar_directory, position_directory, files_lidar_chunck, complete_grid_maps, complete_grid_maps_BB):
    generate_combined_grid_maps_pred(lidar_directory, position_directory, files_lidar_chunck, complete_grid_maps, complete_grid_maps_BB) # type: ignore

    return complete_grid_maps, complete_grid_maps_BB

def generate_chunk(lidar_paths, position_paths, num_chunks, chunk_type):
    
    print(f"\nStarting {chunk_type} set generation")

    all_files = []
    files_for_chunck = []

    for i in range(NUMBER_OF_SENSORS):

        all_files.append(generate_combined_list(sorted([f for f in os.listdir(lidar_paths[i])]),sorted([f for f in os.listdir(position_paths[i])])))

        sum_ped, sum_bic, sum_car = number_of_BB(sorted([f for f in os.listdir(position_paths[i])]), position_paths[i])
        lenght = len(sorted([f for f in os.listdir(position_paths[i])]))
        print(f"\nSum_complete lidar {i+1}: ", sum_ped, sum_bic, sum_car)
        print(f"Average_complete lidar {i+1}: ", sum_ped/lenght, sum_bic/lenght, sum_car/lenght)

    i = 0

    print("\n")
    for i in range(NUMBER_OF_SENSORS):
        files_for_chunck.append(math.ceil(len(all_files[i]) / num_chunks)) #type: ignore
        print(f"Number of files of lidar {i+1} and for each chunk: {len(all_files[i])}    {files_for_chunck[i]}")
    
    i = 0

    gc.collect()

    for i in range(num_chunks):

        print(f"\nChunck number {i+1} of {chunk_type}: ")

        complete_grid_maps = []
        complete_grid_maps_BB = []

        all_files_chunck = []

        for k in range(NUMBER_OF_SENSORS):

            files_lidar_chunck = all_files[k][ i*files_for_chunck[k] : min( (i+1)*files_for_chunck[k], len(all_files[k]) ) ] #type: ignore
            all_files_chunck.append(files_lidar_chunck)
        '''
        with ThreadPoolExecutor(max_workers=NUMBER_OF_SENSORS) as executor:
            futures = []
            for j in range(NUMBER_OF_SENSORS):
                futures.append(executor.submit(process_lidar_chunk, lidar_paths[j], position_paths[j], all_files_chunck[j], complete_grid_maps, complete_grid_maps_BB)) # type: ignore
            
            for future in futures:
                complete_grid_maps, complete_grid_maps_BB = future.result()
        '''
        for j in range (NUMBER_OF_SENSORS): 
            process_lidar_chunk (lidar_paths[j], position_paths[j], all_files_chunck[j], complete_grid_maps, complete_grid_maps_BB)

        print("\n")
        complete_grid_maps = np.array(complete_grid_maps)

        complete_grid_maps_BB = np.array(complete_grid_maps_BB)

        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)
        
        #complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

        if (chunk_type == 'val' or chunk_type == 'test'):
            indices = np.arange(complete_grid_maps.shape[0])
            np.random.shuffle(indices)
            complete_grid_maps = complete_grid_maps[indices]
            complete_grid_maps_BB = complete_grid_maps_BB[indices]

        print(f"complete grid map shape : {complete_grid_maps.shape}")
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_{chunk_type}_{i}.npy'), complete_grid_maps)
        print(f"complete grid map {chunk_type} {i} saved")
        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_{chunk_type}_{i}.npy'), complete_grid_maps_BB)
        print(f"complete grid map BB {chunk_type} {i} saved")

if __name__ == '__main__':
    gc.collect()
    set_start_method("spawn", force=True)
    random.seed(SEED)

    complete_name_chunck_path = os.path.join(CHUNCKS_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    os.makedirs(complete_name_chunck_path, exist_ok=True)

    with open(os.path.join(SCALER_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)

    lidar_train_direcory_list = []
    BB_train_directory_list = []

    lidar_test_direcory_list = []
    BB_test_directory_list = []

    lidar_val_direcory_list = []
    BB_val_directory_list = []

    for i in range (1, NUMBER_OF_SENSORS+1):
        lidar_train_direcory_list.append(LIDAR_X_GRID_DIRECTORY.replace('X', str(i)))
        BB_train_directory_list.append(POSITION_LIDAR_X_GRID.replace('X', str(i)))
        lidar_test_direcory_list.append(LIDAR_X_TEST.replace('X', str(i)))
        BB_test_directory_list.append(POSITION_X_TEST.replace('X', str(i)))
        lidar_val_direcory_list.append(LIDAR_X_VAL.replace('X', str(i)))
        BB_val_directory_list.append(POSITION_X_VAL.replace('X', str(i)))
    
    generate_chunk(lidar_train_direcory_list, BB_train_directory_list, NUMBER_OF_CHUNCKS, 'train')

    generate_chunk(lidar_test_direcory_list, BB_test_directory_list, NUMBER_OF_CHUNCKS_TEST, 'test')

    generate_chunk(lidar_val_direcory_list, BB_val_directory_list, NUMBER_OF_CHUNCKS_TEST, 'val')
