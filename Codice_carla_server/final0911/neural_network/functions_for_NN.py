import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import sys
from sklearn.model_selection import train_test_split
import importlib
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
import gc
import random
from multiprocessing import Pool, set_start_method
from matplotlib.path import Path
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import gc
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
from constants import *

def link_constants():
    # Dynamically construct the path to the data_gen_and_processing folder
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_gen_and_processing_dir = os.path.join(parent_dir, 'data_gen_and_processing')

    # Add the path to the constants module
    sys.path.insert(0, data_gen_and_processing_dir)

    # Print paths for debugging
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Data gen and processing directory: {data_gen_and_processing_dir}")
    
    # Add the path to the constants module
    sys.path.append(data_gen_and_processing_dir)

def load_points_grid_map(csv_file):
    """Load bounding box vertices from a CSV file."""
    points = np.loadtxt(csv_file, delimiter=',', usecols=(0, 1, 2), dtype=float)
    return points

def load_points_grid_map_BB (csv_file):
    """Load bounding box vertices from a CSV file."""
    points = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            coordinates = [ [ float(row[i]), float(row[i+1]), float(row[i+2]) ] for i in range(2, 12, 3)]
            points.append(coordinates)

    np_points = np.array(points)
    return np_points

def generate_combined_grid_maps(grid_map_path, grid_map_BB_path, grid_map_files, grid_map_BB_files, complete_grid_maps, complete_grid_maps_BB, complete_num_BB, bool_value):
    for file, file_BB in zip(grid_map_files, grid_map_BB_files):
        complete_path = os.path.join(grid_map_path, file)
        complete_path_BB = os.path.join(grid_map_BB_path, file_BB)
        #print(f"Loading {file} and {file_BB}...")

        points = load_points_grid_map(complete_path)
        points_BB = load_points_grid_map_BB(complete_path_BB)

        grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float) # type: ignore
        grid_map_recreate_BB = np.full((Y_RANGE, X_RANGE), 0, dtype=float) # type: ignore

        cols, rows, heights = points.T
        grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

        for i in range(len(points_BB)):
            vertices = np.array(points_BB[i])
            height_BB = 1  # Assuming all vertices have the same height
            fill_polygon(grid_map_recreate_BB, vertices, height_BB)
        
        if bool_value:
            num_BB = [0,0,0]
            with open(complete_path_BB, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if row[1] == 'pedestrian':
                        num_BB[0] += 1
                    elif row[1] == 'bicycle':
                        num_BB[1] += 1
                    elif row[1] == 'car':
                        num_BB[2] += 1

            complete_grid_maps.append(grid_map_recreate)
            complete_grid_maps_BB.append(grid_map_recreate_BB)
            complete_num_BB.append(num_BB)
        else:
            complete_grid_maps.append(grid_map_recreate)
            complete_grid_maps_BB.append(grid_map_recreate_BB)

def fill_polygon(grid_map, vertices, height):
    # Create an empty mask with the same shape as the grid map
    mask = np.zeros_like(grid_map, dtype=np.uint8)
    
    # Convert vertices to integer coordinates
    vertices_int = np.array(vertices[:, :2], dtype=np.int32)
    
    # Define different orders to try
    orders = [
        [0, 1, 3, 2],
        [0, 1, 2, 3]
    ]
    
    # Try filling the polygon with different orders of vertices
    for order in orders:
        ordered_vertices = vertices_int[order]
        cv2.fillPoly(mask, [ordered_vertices], 1)
    
    # Set the height for the filled area in the grid map
    grid_map[mask == 1] = height

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def split_data(lidar_data, BB_data, num_BB, size):
    # Split the dataset into a combined training and validation set, and a separate test set using num_BB as stratification
    X_train_val, X_test, y_train_val, y_test, num_BB_train_val, num_BB_test = train_test_split(
        lidar_data, # Samples
        BB_data, # Labels
        num_BB, # Number of number_of_BB
        test_size = size,
        random_state=SEED, # type: ignore
        stratify=num_BB
    )
    return X_train_val, X_test, y_train_val, y_test, num_BB_train_val, num_BB_test

def process_file_number_BB(file, path):
    sum_ped = 0
    sum_bic = 0
    sum_car = 0
    complete_path = os.path.join(path, file)
    with open(complete_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row[1] == 'pedestrian':
                sum_ped += 1
            elif row[1] == 'bicycle':
                sum_bic += 1
            elif row[1] == 'car':
                sum_car += 1
    return sum_ped, sum_bic, sum_car

def number_of_BB(files, path):
    sum_ped = 0
    sum_bic = 0
    sum_car = 0
    for file in files:
        complete_path = os.path.join(path, file)
        with open(complete_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if row[1] == 'pedestrian':
                    sum_ped += 1
                elif row[1] == 'bicycle':
                    sum_bic += 1
                elif row[1] == 'car':
                    sum_car += 1
    
    return sum_ped, sum_bic, sum_car

def rotate_image(image, angle):
    """
    Rotate the given image by the specified angle using OpenCV.

    Parameters:
    - image: numpy array to be rotated.
    - angle: The angle by which to rotate the image.

    Returns:
    - Rotated numpy array.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated

def slide_horizontal(image, shift):
    """
    Slide the given image horizontally by the specified shift using OpenCV.

    Parameters:
    - image: numpy array to be shifted.
    - shift: The number of pixels to shift the image.

    Returns:
    - Shifted numpy array.
    """
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted

def slide_vertical(image, shift):
    """
    Slide the given image vertically by the specified shift using OpenCV.

    Parameters:
    - image: numpy array to be shifted.
    - shift: The number of pixels to shift the image.

    Returns:
    - Shifted numpy array.
    """
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return shifted

def apply_augmentation(grid_maps, grid_maps_BB):

    augmentations = {
        'horizontal_flip': lambda img: cv2.flip(img, 1),
        'vertical_flip': lambda img: cv2.flip(img, 0),
        'rotation': rotate_image,
        'slide_horizontal': slide_horizontal,
        'slide_vertical': slide_vertical
    }

    augmented_grid_maps = []
    augmented_grid_maps_BB = []

    # Set the random seed for consistency
    random.seed(SEED)
    
    for i in range(grid_maps.shape[0]):
        grid_map = grid_maps[i]
        grid_map_BB = grid_maps_BB[i]
        
        applied_augmentations = set()
        for j in range(2):
            while True:
                # Seleziona casualmente un'augmentation
                augmentation_name, augmentation = random.choice(list(augmentations.items()))
                if augmentation_name not in applied_augmentations:
                    applied_augmentations.add(augmentation_name)
                    break

            # Applica la stessa augmentation a entrambe le immagini
            if augmentation_name == 'rotation':
                while True:
                    angle = random.randint(-45, 45)
                    if angle < -30 or angle > 30:
                        break
                grid_map = augmentation(grid_map, angle)
                grid_map_BB = augmentation(grid_map_BB, angle)

            elif augmentation_name in ['slide_horizontal', 'slide_vertical']:
                while True:
                    shift = random.randint(-100, 100)
                    if shift < -50 or shift > 50:
                        break
                grid_map = augmentation(grid_map, shift)
                grid_map_BB = augmentation(grid_map_BB, shift)
            else:
                grid_map = augmentation(grid_map)
                grid_map_BB = augmentation(grid_map_BB)
        
        augmented_grid_maps.append(grid_map)
        augmented_grid_maps_BB.append(grid_map_BB)
    
    return augmented_grid_maps, augmented_grid_maps_BB

def load_array(file_path):
    return np.load(file_path)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self): # Constructor method for the autoencoder
        super(Autoencoder, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x): # The forward method defines the computation that happens when the model is called with input x.
        x = self.encoder(x).contiguous()
        x = self.decoder(x).contiguous()
        return x

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class WeightedCustomLoss(nn.Module):
    def __init__(self, weight=100):
        super(WeightedCustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.weight = weight

    def forward(self, predictions, targets):
        from constants import FLOOR_HEIGHT
        mask = (targets != 0).float()
        masked_predictions = predictions * mask
        masked_targets = targets * mask
        loss = self.mse_loss(masked_predictions, masked_targets)
        
        # Apply weighting to the loss
        weighted_loss = loss * self.weight + self.mse_loss(predictions * (1 - mask), targets * (1 - mask))
        return weighted_loss
    
class LidarDataset(torch.utils.data.Dataset):
    def __init__(self, grid_maps_dir, grid_maps_bb_dir, chunk_index):
        self.grid_maps_dir = grid_maps_dir
        self.grid_maps_bb_dir = grid_maps_bb_dir
        self.chunk_index = chunk_index

        # Load the entire chunk dataset
        with ThreadPoolExecutor(max_workers=2) as executor:
                self.grid_maps, self.grid_maps_bb = executor.map(load_array, [
                    os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{chunk_index}.npy'),
                    os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{chunk_index}.npy')
                ])
        print("shape of grid_maps", self.grid_maps.shape)
        print("shape of grid_maps_bb", self.grid_maps_bb.shape)

    def __len__(self):
        return len(self.grid_maps)

    def __getitem__(self, idx):
        grid_map = self.grid_maps[idx]
        grid_map_bb = self.grid_maps_bb[idx]
        # Ensure the shape is [channels, height, width]
        return torch.from_numpy(grid_map).float(), torch.from_numpy(grid_map_bb).float()