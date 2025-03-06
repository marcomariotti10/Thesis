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
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import pickle
from datetime import datetime
import random
import math
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler


def process_lidar_chunk(lidar_directory, position_directory, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, is_training):
    generate_combined_grid_maps(lidar_directory, position_directory, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, is_training) # type: ignore

    # Info about the number of bounding boxes
    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, position_directory)
    print(f"\nSum_chunck: ", sum_ped, sum_bic, sum_car)
    print(f"Average_chunck: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

    return complete_grid_maps, complete_grid_maps_BB

def generate_chunk(lidar_paths, position_paths, num_chunks, chunk_type):
    print(f"\nStarting {chunk_type} set generation")

    # Shuffle files_lidar and files_BB in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(lidar_paths[0])]), sorted([f for f in os.listdir(position_paths[0])])))
    random.shuffle(combined_files)
    files_lidar_1, files_BB_1 = zip(*combined_files)
    files_lidar_1 = list(files_lidar_1)
    files_BB_1 = list(files_BB_1)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, position_paths[0])
    print(f"\nSum_complete_{chunk_type}_lidar1: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_{chunk_type}_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))

    combined_files = list(zip(sorted([f for f in os.listdir(lidar_paths[1])]), sorted([f for f in os.listdir(position_paths[1])])))
    random.shuffle(combined_files)
    files_lidar_2, files_BB_2 = zip(*combined_files)
    files_lidar_2 = list(files_lidar_2)
    files_BB_2 = list(files_BB_2)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_2, position_paths[1])
    print(f"\nSum_complete_{chunk_type}_lidar2: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_{chunk_type}_lidar2: ", sum_ped/len(files_BB_2), sum_bic/len(files_BB_2), sum_car/len(files_BB_2))

    combined_files = list(zip(sorted([f for f in os.listdir(lidar_paths[2])]), sorted([f for f in os.listdir(position_paths[2])])))
    random.shuffle(combined_files)
    files_lidar_3, files_BB_3 = zip(*combined_files)
    files_lidar_3 = list(files_lidar_3)
    files_BB_3 = list(files_BB_3)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, position_paths[2])
    print(f"\nSum_complete_{chunk_type}_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_{chunk_type}_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    file_for_chunck1 = math.ceil(total_num_of_files1 / num_chunks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / num_chunks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / num_chunks) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    gc.collect()

    for i in range(num_chunks):

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
            futures.append(executor.submit(process_lidar_chunk, lidar_paths[0], position_paths[0], files_lidar_chunck_1, files_BB_chunck_1, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False)) # type: ignore
            futures.append(executor.submit(process_lidar_chunk, lidar_paths[1], position_paths[1], files_lidar_chunck_2, files_BB_chunck_2, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False)) # type: ignore
            futures.append(executor.submit(process_lidar_chunk, lidar_paths[2], position_paths[2], files_lidar_chunck_3, files_BB_chunck_3, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False)) # type: ignore
            
            for future in futures:
                complete_grid_maps, complete_grid_maps_BB = future.result()

        complete_grid_maps = np.array(complete_grid_maps)
        print(f"complete grid map shape : {complete_grid_maps.shape}")

        complete_grid_maps_BB = np.array(complete_grid_maps_BB)
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)
        
        complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

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
    with open(os.path.join(SCALER_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    generate_chunk([LIDAR_1_GRID_DIRECTORY, LIDAR_2_GRID_DIRECTORY, LIDAR_3_GRID_DIRECTORY], 
                   [POSITION_LIDAR_1_GRID_NO_BB, POSITION_LIDAR_2_GRID_NO_BB, POSITION_LIDAR_3_GRID_NO_BB], 
                   NUMBER_OF_CHUNCKS, 'train')

    generate_chunk([LIDAR_1_TEST, LIDAR_2_TEST, LIDAR_3_TEST], 
                   [POSITION_1_TEST, POSITION_2_TEST, POSITION_3_TEST], 
                   NUMBER_OF_CHUNCKS_TEST, 'test')

    generate_chunk([LIDAR_1_VAL, LIDAR_2_VAL, LIDAR_3_VAL], 
                   [POSITION_1_VAL, POSITION_2_VAL, POSITION_3_VAL], 
                   NUMBER_OF_CHUNCKS_VAL, 'val')
