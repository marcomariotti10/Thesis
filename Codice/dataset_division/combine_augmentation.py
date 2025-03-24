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

    random.seed(SEED)

    complete_name_chunck_path = os.path.join(CHUNCKS_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')

    os.makedirs(complete_name_chunck_path, exist_ok=True)

    train_list = random.sample(range(NUMBER_OF_CHUNCKS), NUMBER_OF_CHUNCKS)
    augment_list = random.sample(range(NUMBER_OF_CHUNCKS), NUMBER_OF_CHUNCKS)

    print("Train list: ", train_list)
    print("Augment list: ", augment_list)

    for i in range (NUMBER_OF_CHUNCKS):
        
        train_chunck = train_list[i]
        augment_chunck = augment_list[i]

        with ThreadPoolExecutor(max_workers=4) as executor:
            complete_grid_maps , complete_grid_maps_BB, augmented_grid_maps, augmented_grid_maps_BB = executor.map(load_array, [
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_train_{train_chunck}.npy'),
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_train_{train_chunck}.npy'),
                os.path.join(complete_name_chunck_path, f'augmented_grid_maps_train_{augment_chunck}.npy'),
                os.path.join(complete_name_chunck_path, f'augmented_grid_maps_BB_train_{augment_chunck}.npy')
            ])
        

        print("Lenght complete grid maps: ", len(complete_grid_maps))
        print("Lenght complete grid maps BB: ", len(complete_grid_maps_BB))
        print("Lenght augmented grid maps: ", len(augmented_grid_maps))
        print("Lenght augmented grid maps BB: ", len(augmented_grid_maps_BB))

        complete_grid_maps = np.concatenate((complete_grid_maps, augmented_grid_maps), axis=0)
        complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)

        indices = np.arange(complete_grid_maps.shape[0])
        np.random.shuffle(indices)
        complete_grid_maps = complete_grid_maps[indices]
        complete_grid_maps_BB = complete_grid_maps_BB[indices]

        print("Lenght complete grid maps aufter concatenation: ", len(complete_grid_maps))
        print("Lenght complete grid maps BB after concatenation: ", len(complete_grid_maps_BB))

        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_train_{train_chunck}.npy'), complete_grid_maps)
        print(f"Saved complete_grid_maps_train_{train_chunck}.npy")
        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_train_{train_chunck}.npy'), complete_grid_maps_BB)
        print(f"Saved complete_grid_maps_BB_train_{train_chunck}.npy")

        os.remove(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_train_{augment_chunck}.npy'))
        os.remove(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_BB_train_{augment_chunck}.npy'))