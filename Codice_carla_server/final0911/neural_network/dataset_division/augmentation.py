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

if __name__ == '__main__':

    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *

    gc.collect()

    set_start_method("spawn", force=True)

    random.seed(SEED)

    os.makedirs(CHUNCKS_DIR, exist_ok=True)

    for i in range (NUMBER_OF_CHUNCKS):
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            complete_grid_maps , complete_grid_maps_BB = executor.map(load_array, [
                os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{i}.npy'),
                os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{i}.npy')
            ])

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

        print("shape after expand_dims: ", complete_grid_maps.shape, complete_grid_maps_BB.shape)

        # Split the data
        split_index = math.ceil(len(complete_grid_maps) * 0.9)
        X_val = complete_grid_maps[split_index:]
        complete_grid_maps = complete_grid_maps[:split_index]
        y_val = complete_grid_maps_BB[split_index:]
        complete_grid_maps_BB = complete_grid_maps_BB[:split_index]


        # Save the arrays
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_train_{i}.npy'), complete_grid_maps)
        print(f"complete grid map train {i} saved")
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_train_{i}.npy'), complete_grid_maps_BB)
        print(f"complete grid map BB train {i} saved")

        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_val_{i}.npy'), X_val)
        print(f"complete grid map train {i} saved")
        np.save(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_val_{i}.npy'), y_val)
        print(f"complete grid map BB train {i} saved")

        os.remove(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{i}.npy'))
        os.remove(os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{i}.npy'))
