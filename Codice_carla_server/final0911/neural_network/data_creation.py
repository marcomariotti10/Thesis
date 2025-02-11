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

def import_constants():
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
    points = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            coordinates = [
                float(row[0]), float(row[1]), float(row[2])
                ]
            points.append(coordinates)
    np_points = np.array(points)
    return np_points

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

def process_combined_file(file, file_BB, grid_map_path, grid_map_BB_path):
    try:
        # Import constants inside the function
        from constants import Y_RANGE, X_RANGE, FLOOR_HEIGHT

        complete_path = os.path.join(grid_map_path, file)
        complete_path_BB = os.path.join(grid_map_BB_path, file_BB)
        print(f"Loading {file} and {file_BB}...")

        points = load_points_grid_map(complete_path)
        points_BB = load_points_grid_map_BB(complete_path_BB)

        num_BB = points_BB.shape[0]

        grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=int)
        grid_map_recreate_BB = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=int)

        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = int(height)

        for i in range(len(points_BB)):
            for j in range(4):
                col, row, height = points_BB[i][j]
                grid_map_recreate_BB[int(row), int(col)] = int(height)

        return grid_map_recreate, grid_map_recreate_BB, num_BB
    except Exception as e:
        print(f"Error processing files {file} and {file_BB}: {e}")
        return None, None, None

def generate_combined_grid_maps(grid_map_path, grid_map_BB_path, complete_grid_maps, complete_grid_maps_BB, complete_num_BB):
    grid_map_files = sorted([f for f in os.listdir(grid_map_path)])
    grid_map_BB_files = sorted([f for f in os.listdir(grid_map_BB_path)])
    
    with Pool() as pool:
        results = pool.starmap(process_combined_file, [(file, file_BB, grid_map_path, grid_map_BB_path) for file, file_BB in zip(grid_map_files, grid_map_BB_files)])
    
    for gm, gmbb, nb in results:
        if gm is not None and gmbb is not None and nb is not None:
            complete_grid_maps.append(gm)
            complete_grid_maps_BB.append(gmbb)
            complete_num_BB.append(nb)
        else:
            print("Error processing file.")

def split_data(lidar_data, BB_data, num_BB, size):
    # Split the dataset into a combined training and validation set, and a separate test set using num_BB as stratification
    X_train_val, X_test, y_train_val, y_test, num_BB_train_val, num_BB_test = train_test_split(
        lidar_data, # Samples
        BB_data, # Labels
        num_BB, # Number of BB
        test_size = size,
        random_state=SEED, # type: ignore
        stratify=num_BB
    )
    return X_train_val, X_test, y_train_val, y_test, num_BB_train_val, num_BB_test

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self): # Constructor method for the autoencoder
        super(Autoencoder, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
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
        x = self.encoder(x)
        x = self.decoder(x)
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

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
