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

if __name__ == '__main__':

    gc.collect()

    set_start_method("spawn", force=True)

    random.seed(SEED)

    complete_name_chunck_path = os.path.join(CHUNCKS_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')

    os.makedirs(complete_name_chunck_path, exist_ok=True)

    for i in range (NUMBER_OF_CHUNCKS):
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            complete_grid_maps , complete_grid_maps_BB = executor.map(load_array, [
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_train_{i}.npy'),
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_train_{i}.npy')
            ])

        
        # Number of random samples you want to take
        num_samples = int(complete_grid_maps.shape[0] * AUGMENTATION_FACTOR)

        # Generate random indices
        random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

        # Select elements at the random indices
        random_complete_grid_maps = complete_grid_maps[random_indices]
        random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]

        del complete_grid_maps, complete_grid_maps_BB

        print(f"\nRandom complete grid maps shape: {random_complete_grid_maps.shape}")
        print(f"Random complete grid maps BB shape: {random_complete_grid_maps_BB.shape}")

        augmented_grid_maps, augmented_grid_maps_BB = apply_augmentation(random_complete_grid_maps, random_complete_grid_maps_BB)

        del random_complete_grid_maps, random_complete_grid_maps_BB
        
        augmented_grid_maps = np.array(augmented_grid_maps)
        augmented_grid_maps_BB = np.array(augmented_grid_maps_BB)

        print(f"\nAugmented grid maps shape: {augmented_grid_maps.shape}")
        print(f"Augmented grid maps BB shape: {augmented_grid_maps_BB.shape}")

        # Save the arrays
        np.save(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_train_{i}.npy'), augmented_grid_maps)
        print(f"augmented grid map train {i} saved")
        np.save(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_BB_train_{i}.npy'), augmented_grid_maps_BB)
        print(f"augmented grid map BB train {i} saved")

            