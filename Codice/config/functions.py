import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import sys
from sklearn.model_selection import train_test_split
import importlib
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
import math
import ast
import gc
from multiprocessing import Pool
from sklearn.preprocessing import MinMaxScaler
import platform

if platform.system() in 'Linux':
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, RandomHorizontalFlip, RandomTranslate, Cutout
    from ffcv.fields.decoders import NDArrayDecoder

############################################
#        LOADING DATA FUNCTION             #
############################################

def load_points_grid_map(csv_file):
    """
    Load 3D point cloud data from a CSV file for grid map reconstruction.
    
    This function reads a CSV file containing 3D coordinates (x, y, z) representing
    LiDAR point cloud data that will be used to create occupancy grid maps.
    
    Args:
        csv_file (str): Path to the CSV file containing point cloud data
        
    Returns:
        np.ndarray: Array of shape (n_points, 3) containing x, y, z coordinates
    """
    points = np.loadtxt(csv_file, delimiter=',', usecols=(0, 1, 2), dtype=float)
    return points

def load_points_grid_map_BB(csv_file):
    """
    Load bounding box data from a CSV file for object detection and tracking.
    
    This function reads bounding box information from a CSV file, extracting
    coordinate pairs that define bounding boxes for detected objects (pedestrians,
    vehicles, etc.) in the scene.
    
    Args:
        csv_file (str): Path to the CSV file containing bounding box data
        
    Returns:
        tuple: A tuple containing:
            - np_points (np.ndarray): Array containing bounding box coordinates as strings
            - indeces (np.ndarray): Array containing object IDs/indices
    """
    points = []
    indeces = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            coordinates = [row[2]]
            points.append(coordinates)
            indeces.append(row[0])

    np_points = np.array(points)
    indeces = np.array(indeces)
    return np_points, indeces

# Generate each pair of grid map and bounding box map to be used for the partial fit
def generate_combined_grid_maps_fit(grid_map_path, grid_map_files, complete_grid_maps):
    """
    Generate grid maps for fitting data into scaler.
    
    This function processes LiDAR point cloud files to create 2D occupancy grid maps
    that represent the environment structure. Each grid map is reconstructed from
    3D point cloud data by projecting points onto a 2D grid with height information.
    
    Args:
        grid_map_path (str): Path to the directory containing grid map files
        grid_map_files (list): List of filenames to process
        complete_grid_maps (list): Output list to store the generated grid maps
        
    Returns:
        None: Modifies complete_grid_maps list in place
    """
    
    for file in grid_map_files:
        complete_path = os.path.join(grid_map_path, file)
        #print(f"Loading {file} and {file_BB}...")

        points = load_points_grid_map(complete_path)
            
        grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float) # type: ignore

        cols, rows, heights = points.T
        grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

        complete_grid_maps.append(grid_map_recreate)

def generate_combined_grid_maps_pred(grid_map_path, grid_map_BB_path, grid_map_files, complete_grid_maps, complete_grid_maps_BB):
    """
    Generate combined grid maps for prediction tasks with temporal sequences.
    
    This function creates sequences of grid maps and corresponding bounding box maps
    for training predictive models. It processes both environmental structure data
    (LiDAR) and object detection data (bounding boxes) to create input-output pairs
    for temporal prediction tasks.
    
    Args:
        grid_map_path (str): Path to directory containing LiDAR grid map files
        grid_map_BB_path (str): Path to directory containing bounding box files  
        grid_map_files (list): List of file sequences for processing
        complete_grid_maps (list): Output list for environmental grid map sequences
        complete_grid_maps_BB (list): Output list for bounding box grid map sequences
        
    Returns:
        None: Modifies complete_grid_maps and complete_grid_maps_BB lists in place
    """
    
    #print(len(grid_map_files))
    
    for i in range(len(grid_map_files)):

        actors = {}
        
        grid_map_group = []

        # Process input sequence of environmental grid maps
        for j in range (NUMBER_RILEVATIONS_INPUT):

            complete_path = os.path.join(grid_map_path, grid_map_files[i][j])

            points = load_points_grid_map(complete_path)

            grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float) # type: ignore

            cols, rows, heights = points.T
            grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

            grid_map_group.append(grid_map_recreate)

            # Count actor appearances for filtering persistent objects
            complete_path_BB = os.path.join(grid_map_BB_path, grid_map_files[i][NUMBER_RILEVATIONS_INPUT + j])

            with open(complete_path_BB, 'r') as file_BB:
                reader = csv.reader(file_BB)
                next(reader)  # Skip header
                for row in reader: 
                    if row[3] == 'yes':
                        if row[0] not in actors:
                            actors[row[0]] = 1
                        else:
                            actors[row[0]] += 1

        complete_grid_maps.append(grid_map_group)

        grid_map_BB_group = []

        # Process future target bounding box maps
        for k in range(len(FUTURE_TARGET_RILEVATION)):   

            complete_path_BB = os.path.join(grid_map_BB_path, grid_map_files[i][-len(FUTURE_TARGET_RILEVATION) + k])

            points_BB, indeces = load_points_grid_map_BB(complete_path_BB)

            all_pairs = []

            # Only include actors that appear frequently enough (persistent objects)
            for idx, row in enumerate(points_BB):
                if indeces[idx] in actors and actors[indeces[idx]] >= MINIMUM_NUMBER_OF_RILEVATIONS:
                    # Extract the string from the array
                    string_data = row[0]
                    # Safely evaluate the string to convert it into a list of tuples
                    pairs = ast.literal_eval(string_data)
                    # Add the pairs to the all_pairs list
                    all_pairs.extend(pairs)
                
            all_pairs = np.array(all_pairs)

            grid_map_recreate_BB = np.full((Y_RANGE, X_RANGE), 0, dtype=float) # type: ignore

            if len(all_pairs) != 0:
                cols, rows = all_pairs.T
                grid_map_recreate_BB[rows.astype(int), cols.astype(int)] = 1

            grid_map_BB_group.append(grid_map_recreate_BB)

        complete_grid_maps_BB.append(grid_map_BB_group)

def generate_combined_list(files_lidar, files_BB, type):
    """
    Generate a new list where each element contains NUMBER_RILEVATIONS_INPUT elements from files_lidar
    and 1 element from files_BB, which is FUTURE_TARGET_RILEVATION positions ahead of the last lidar element.

    Parameters:
    - files_lidar: List of lists containing sorted lidar file names.
    - files_BB: List of lists containing sorted bounding box file names.
    - NUMBER_RILEVATIONS_INPUT: Number of lidar files to include in each group.
    - FUTURE_TARGET_RILEVATION: Number of positions ahead for the bounding box file.

    Returns:
    - combined_list: A list where each element contains NUMBER_RILEVATIONS_INPUT lidar files and 1 bounding box file.
    """
    combined_list = []
    #Can't use match because not supported by python 3.9
    if type == 'train':
        start =  int(len(files_lidar)*TEST_SIZE)
        finish = int(len(files_lidar)*TEST_SIZE*9)
    elif type == 'val':
        start = int(len(files_lidar)*TEST_SIZE*9)
        finish = int(len(files_lidar))
    elif type == 'test':
        start = 0
        finish = int(len(files_lidar)*TEST_SIZE)

    #print("\nlen files_lidar:", len(files_lidar))
    for i in range(start, finish - NUMBER_RILEVATIONS_INPUT - FUTURE_TARGET_RILEVATION[-1] + 1, NUMBER_RILEVATIONS_INPUT):
        # Take NUMBER_RILEVATIONS_INPUT lidar files starting from the current index
        lidar_group = files_lidar[i:i + NUMBER_RILEVATIONS_INPUT]
        # Take the FUTURE_TARGET_RILEVATION-th BB file after the last lidar file
        BB_files = files_BB[i:i + NUMBER_RILEVATIONS_INPUT]
        BB_file_future = []
        for j in range(len(FUTURE_TARGET_RILEVATION)):
            BB_file_future.append(files_BB[i + NUMBER_RILEVATIONS_INPUT + FUTURE_TARGET_RILEVATION[j] - 1])
        # Combine the lidar group and the BB file into one element
        combined_list.append(lidar_group + BB_files + BB_file_future)

    random.shuffle(combined_list)
    return combined_list

def fill_polygon(grid_map, vertices, height):
    """
    Fill a polygonal area in a grid map with a specified height value.
    
    This function takes a set of vertices defining a polygon and fills the
    corresponding area in the grid map with the given height value. It tries
    different vertex orderings to handle various polygon orientations.
    
    Args:
        grid_map (np.ndarray): 2D grid map to modify
        vertices (np.ndarray): Array of polygon vertices (x, y coordinates)
        height (float): Height value to fill the polygon area with
        
    Returns:
        None: Modifies grid_map in place
    """
    # Create an empty mask with the same shape as the grid map
    mask = np.zeros_like(grid_map, dtype=np.uint8)
    
    # Convert vertices to integer coordinates
    vertices_int = np.array(vertices[:, :2], dtype=np.int32)
    #print("vertices_int", vertices_int)
    
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


############################################
#        DATA AUGMENTATION FUNCTION        #
############################################

def apply_augmentation(random_gm, random_BB):
    """
    Apply data augmentation techniques to grid map and bounding box data.
    
    This function performs random data augmentation on grid maps and their corresponding
    bounding box annotations. It applies combinations of rotation, shifting, and flipping
    to increase data diversity for training neural networks.
    
    Args:
        random_gm (np.ndarray): Array of grid maps to augment
        random_BB (np.ndarray): Array of corresponding bounding box maps
        
    Returns:
        tuple: A tuple containing:
            - augmented_grid_maps (list): List of augmented grid maps
            - augmented_grid_maps_BB (list): List of augmented bounding box maps
    """
    def random_shift(img, axis, shift):
        """
        Apply a shift to the image along the specified axis and pad with zeros.
        """
        shifted_img = np.roll(img, shift=shift, axis=axis)
        if axis == 1:  # Horizontal shift
            if shift > 0:  # Positive shift (right)
                shifted_img[:, :shift] = 0
            else:  # Negative shift (left)
                shifted_img[:, shift:] = 0
        elif axis == 0:  # Vertical shift
            if shift > 0:  # Positive shift (down)
                shifted_img[:shift, :] = 0
            else:  # Negative shift (up)
                shifted_img[shift:, :] = 0
        return shifted_img

    def random_rotation(img, angle):
        """
        Rotate an image by a specified angle.
        """
        return rotate_image(img, angle)

    augmented_grid_maps = []
    augmented_grid_maps_BB = []

    # Set the random seed for consistency
    random.seed(SEED)
    
    for i in range(random_gm.shape[0]):
        # Create copies to avoid modifying the original arrays
        grid_map = np.copy(random_gm[i])
        grid_map_BB = np.copy(random_BB[i])

        # Define augmentation probabilities
        augmentations = [
            ('rotation', 0.5),
            ('shift', 0.2),
            ('flip', 0.3)
        ]

        # Select the first augmentation
        first_augmentation = random.choices(
            augmentations, 
            weights=[aug[1] for aug in augmentations], 
            k=1
        )[0]

        # Remove the first augmentation from the list for the second selection
        remaining_augmentations = [aug for aug in augmentations if aug[0] != first_augmentation[0]]

        # Select the second augmentation
        second_augmentation = random.choices(
            remaining_augmentations, 
            weights=[aug[1] for aug in remaining_augmentations], 
            k=1
        )[0]

        # Apply the selected augmentations
        for augmentation_type in [first_augmentation, second_augmentation]:
            if augmentation_type[0] == 'rotation':
                angle = int(random.uniform(-60, -30)) if random.random() < 0.5 else int(random.uniform(30, 60))
                #print("angle", angle)
                augmentation = lambda img: random_rotation(img, angle=angle)
            elif augmentation_type[0] == 'shift':
                shift = int(random.uniform(-100, -50)) if random.random() < 0.5 else int(random.uniform(50, 100))
                #print("shift", shift)
                axis = random.choice([0, 1])  # Randomly choose vertical or horizontal shift
                #print("axis", axis)
                augmentation = lambda img: random_shift(img, axis=axis, shift=shift)
            elif augmentation_type[0] == 'flip':
                flip_code = random.choice([0, 1])  # 0 for vertical flip, 1 for horizontal flip
                #print("flip_code", flip_code)
                augmentation = lambda img: cv2.flip(img, flip_code)

            # Apply the augmentation to all images in the group
            for k in range(grid_map.shape[0]):
                grid_map[k] = augmentation(grid_map[k])
            for z in range(grid_map_BB.shape[0]):
                grid_map_BB[z] = augmentation(grid_map_BB[z])

        augmented_grid_maps.append(grid_map)
        augmented_grid_maps_BB.append(grid_map_BB)

    return augmented_grid_maps, augmented_grid_maps_BB


def rotate_image(image, angle):
    """
    Rotate an image by a specified angle using OpenCV.
    
    This function rotates a 2D image around its center by the given angle.
    It uses nearest neighbor interpolation to preserve discrete values
    in grid maps and sets border values to 0.
    
    Args:
        image (np.ndarray): 2D image array to rotate
        angle (float): Rotation angle in degrees (positive = clockwise)
        
    Returns:
        np.ndarray: Rotated image with same dimensions as input
    """
    # Get the center of the image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated_image

############################################
#       GENERATE CHUNCKS FUNCTIONS         #
############################################

def load_array(file_path):
    """
    Load a NumPy array from a file.
    
    This is a simple wrapper function for np.load() to load
    pre-saved NumPy arrays from disk.
    
    Args:
        file_path (str): Path to the .npy file to load
        
    Returns:
        np.ndarray: Loaded NumPy array
    """
    return np.load(file_path)

def number_of_BB(files, path):
    """
    Count the number of different object types (bounding boxes) in CSV files.
    
    This function analyzes bounding box data files to count occurrences of
    different object types: pedestrians, bicycles, and cars. This is useful
    for dataset analysis and class distribution statistics.
    
    Args:
        files (list): List of CSV filenames to process
        path (str): Directory path containing the CSV files
        
    Returns:
        tuple: A tuple containing counts of:
            - sum_ped (int): Number of pedestrian instances
            - sum_bic (int): Number of bicycle instances  
            - sum_car (int): Number of car instances
    """
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

############################################
#       NEURAL NETWORK FUNCTIONS           #
############################################

def load_dataset(name, i, device, batch):
    """
    Load a dataset using FFCV (Fast Forward Computer Vision) format.
    
    This function creates a high-performance data loader for training neural networks
    using FFCV format. It configures data loading pipelines with appropriate
    transformations and device placement for efficient GPU training.
    
    Args:
        name (str): Dataset type ('train', 'val', or 'test')
        i (int): Dataset index/chunk number
        device (torch.device): Target device for data loading (CPU/GPU)
        batch (int): Batch size for data loading
        
    Returns:
        ffcv.Loader: Configured FFCV data loader with preprocessing pipelines
    """
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_name_ffcv_path = os.path.join(FFCV_DIR)
    complete_path_train = os.path.join(complete_name_ffcv_path, name_train)

    random_seed = random.randint(0, 1000)

    if name == 'train':
        order_op = OrderOption.RANDOM
    else:
        order_op = OrderOption.SEQUENTIAL
    
    train_loader = Loader(complete_path_train, batch_size=batch,
    num_workers=8, order=order_op, distributed=True, seed = random_seed, drop_last= True,
    os_cache=False,
    pipelines={
        'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                    ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                    ToDevice(device, non_blocking=True)],
        'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                ToDevice(device, non_blocking=True)]
    })

    return train_loader

def initialize_weights(m):
    """Applies weight initialization to the model layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  
        init.kaiming_normal_(m.weight)  
        if m.bias is not None:
            init.constant_(m.bias, 0)  
    elif isinstance(m, nn.Linear):  
        init.kaiming_normal_(m.weight)  
        if m.bias is not None:
            init.constant_(m.bias, 0)  
    elif isinstance(m, nn.BatchNorm2d):  
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

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

def calculate_dead_neuron(model, device):
    """
    Calculate the percentage of dead neurons in autoencoder layers.
    
    This function analyzes a trained autoencoder model to identify neurons
    that consistently output zero values (dead neurons). It processes both
    encoder and decoder layers separately and provides statistics.
    
    Args:
        model (nn.Module): Autoencoder model to analyze
        device (torch.device): Device for computation (CPU/GPU)
        
    Returns:
        None: Prints dead neuron percentages for each layer
    """
    # Function to calculate dead neuron percentage
    def dead_neuron_percentage(activations):
        # activations: (batch_size, num_neurons, height, width)
        print(activations.shape)
        num_neurons = activations.shape[1] * activations.shape[2] * activations.shape[3]
        # For each neuron, check if it was always zero across the batch
        dead_neurons = (activations == 0).all(dim=(0, 2, 3)).sum().item()
        return 100.0 * dead_neurons / activations.shape[1]
    
    # Get the first batch of data
    loader = load_dataset('train', 0, device, 16)
    first_batch = next(iter(loader))

    # Unpack the inputs and targets from the first batch
    x, targets = first_batch

    with torch.no_grad():
        # Dynamically compute encoder activations
        encoder_activations = []
        activation = x
        for layer in model.encoder:
            activation = layer(activation)
            encoder_activations.append(activation)

        # Dynamically compute decoder activations
        decoder_activations = []
        activation = encoder_activations[-1]  # Start with the last encoder activation
        for layer in model.decoder:
            activation = layer(activation)
            decoder_activations.append(activation)

    # Calculate and print dead neuron percentages for encoder and decoder layers
    print("ENCODER LAYERS\n")
    for i, e_activation in enumerate(encoder_activations):
        print(f"Dead neurons in encoder layer {i + 1}: {dead_neuron_percentage(e_activation):.2f}%")

    print("\nDECODER LAYERS\n")
    for i, d_activation in enumerate(decoder_activations):
        print(f"Dead neurons in decoder layer {i + 1}: {dead_neuron_percentage(d_activation):.2f}%")

def calculate_dead_neuron_multi(model, device):
    """
    Calculate dead neuron percentages for multi-head autoencoder models.
    
    This function extends dead neuron analysis to multi-head architectures
    where multiple decoder heads share a common encoder. It analyzes the
    shared encoder and each decoder head separately.
    
    Args:
        model (nn.Module): Multi-head autoencoder model to analyze
        device (torch.device): Device for computation (CPU/GPU)
        
    Returns:
        None: Prints dead neuron percentages for encoder and each decoder head
    """
    # Function to calculate dead neuron percentage
    def dead_neuron_percentage(activations):
        # activations: (batch_size, num_neurons, height, width)
        num_neurons = activations.shape[1] * activations.shape[2] * activations.shape[3]
        # For each neuron, check if it was always zero across the batch
        dead_neurons = (activations == 0).all(dim=(0, 2, 3)).sum().item()
        return 100.0 * dead_neurons / num_neurons

    # Get the first batch of data
    loader = load_dataset('train', 0, device, 16)
    first_batch = next(iter(loader))

    # Unpack the inputs and targets from the first batch
    x, targets = first_batch

    with torch.no_grad():
        # Dynamically compute encoder activations
        encoder_activations = []
        activation = x
        for layer in model.encoder:
            activation = layer(activation)
            encoder_activations.append(activation)

        # Dynamically compute decoder activations for each head
        decoder_activations = []
        for decoder in model.decoder:  # Iterate over each decoder head
            activation = encoder_activations[-1]  # Start with the last encoder activation
            head_activations = []
            for layer in decoder:
                activation = layer(activation)
                head_activations.append(activation)
            decoder_activations.append(head_activations)

    # Calculate and print dead neuron percentages for encoder layers
    print("ENCODER LAYERS\n")
    for i, e_activation in enumerate(encoder_activations):
        print(f"Dead neurons in encoder layer {i + 1}: {dead_neuron_percentage(e_activation):.2f}%")

    # Calculate and print dead neuron percentages for each decoder head
    print("\nDECODER LAYERS\n")
    for head_idx, head_activations in enumerate(decoder_activations):
        print(f"Decoder Head {head_idx + 1}:\n")
        for i, d_activation in enumerate(head_activations):
            print(f"  Dead neurons in decoder layer {i + 1}: {dead_neuron_percentage(d_activation):.2f}%")



############################################
#            MODELS SINGLE HEAD            #
############################################

class Autoencoder(nn.Module):
    def __init__(self, activation_fn=nn.ReLU): # Constructor method for the autoencoder
        super(Autoencoder, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_RILEVATIONS_INPUT, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x): # The forward method defines the computation that happens when the model is called with input x.
        x = self.encoder(x).contiguous()
        x = self.decoder(x).contiguous()
        return x

class DoubleConv(nn.Module):
    """Two convolutional layers with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, activation_fn):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            activation_fn()
        )
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, input_channels=7, output_channels=1, features=[16, 32, 64, 128], activation_fn=nn.ReLU):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(input_channels, feature, activation_fn))
            input_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, activation_fn)

        # Decoder path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature, activation_fn))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], output_channels, kernel_size=1)

    def forward(self, x_t, past_frames, timestep):
        x = torch.cat([x_t, past_frames, timestep], dim=1)  # Concatenate inputs along the channel dimension
        skip_connections = []

        # Encoder forward pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward pass
        skip_connections = skip_connections[::-1]  # Reverse skip connections
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Transposed convolution (upsampling)
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = self._crop(skip_connection, x)  # Crop skip connection to match the size of x
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[i + 1](x)  # DoubleConv layer

        return self.final_conv(x)  # Final output layer

    def _crop(self, skip, x):
        """Crop the skip connection to match the size of x."""
        _, _, h, w = x.shape
        skip = torchvision.transforms.functional.center_crop(skip, [h, w])
        return skip
    
class ChannelAttention(nn.Module):
    """
    Channel Attention Module from CBAM (Convolutional Block Attention Module).
    
    This module computes attention weights for each channel in the feature map
    by using both average pooling and max pooling operations followed by a
    shared MLP. It helps the network focus on the most informative channels.
    
    Args:
        in_planes (int): Number of input channels
        reduction (int): Reduction ratio for the MLP bottleneck (default: 16)
    """
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module from CBAM (Convolutional Block Attention Module).
    
    This module computes attention weights for each spatial location in the
    feature map by using channel-wise average and max pooling followed by
    a convolutional layer. It helps the network focus on important spatial regions.
    
    Args:
        kernel_size (int): Size of the convolutional kernel (default: 7, must be 3 or 7)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(x_cat))
        return attention * x

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    CBAM is an attention mechanism that combines both channel and spatial attention
    to improve feature representation. It sequentially applies channel attention
    followed by spatial attention to refine feature maps.
    
    Args:
        in_planes (int): Number of input channels
        reduction (int): Reduction ratio for channel attention (default: 16)
        kernel_size (int): Kernel size for spatial attention (default: 7)
    """
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class Autoencoder_CBAM(nn.Module):
    def __init__(self, activation_fn=nn.ReLU):
        super(Autoencoder_CBAM, self).__init__()
        # Encoder definitions
        self.enc_conv1 = nn.Conv2d(NUMBER_RILEVATIONS_INPUT, 16, kernel_size=3, padding=1)
        self.enc_act1 = activation_fn()
        self.cbam1 = CBAM(16)

        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_act2 = activation_fn()
        self.cbam2 = CBAM(32)

        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_act3 = activation_fn()
        self.pool1    = nn.MaxPool2d(2, stride=2)
        self.cbam3 = CBAM(64)

        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_act4 = activation_fn()
        self.pool2    = nn.MaxPool2d(2, stride=2)
        self.cbam4 = CBAM(128)

        self.enc_conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_act5 = activation_fn()
        self.pool3    = nn.MaxPool2d(2, stride=2)
        self.cbam5 = CBAM(256)

        self.enc_conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc_act6 = activation_fn()
        self.pool4    = nn.MaxPool2d(2, stride=2)
        self.cbam6 = CBAM(512)

        # Decoder definitions
        self.dec_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_act1 = activation_fn()
        self.cbam7 = CBAM(512)

        self.dec_conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_act2 = activation_fn()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.cbam8 = CBAM(256)

        self.dec_conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_act3 = activation_fn()
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.cbam9 = CBAM(128)

        self.dec_conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_act4 = activation_fn()
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.cbam10 = CBAM(64)

        self.dec_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_act5 = activation_fn()
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.cbam11 = CBAM(32)

        self.dec_conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_act6 = activation_fn()
        self.cbam12 = CBAM(16)

        self.dec_conv7 = nn.Conv2d(16, 1, kernel_size=3, padding=1)  # final output

    def forward(self, x):
        # Encoder forward
        x = self.enc_act1(self.enc_conv1(x))
        x = self.cbam1(x)

        x = self.enc_act2(self.enc_conv2(x))
        x = self.cbam2(x)

        x = self.enc_act3(self.enc_conv3(x))
        x = self.cbam3(x)
        x = self.pool1(x)

        x = self.enc_act4(self.enc_conv4(x))
        x = self.cbam4(x)
        x = self.pool2(x)

        x = self.enc_act5(self.enc_conv5(x))
        x = self.cbam5(x)
        x = self.pool3(x)

        x = self.enc_act6(self.enc_conv6(x))
        x = self.cbam6(x)
        x = self.pool4(x)

        # Decoder forward
        x = self.dec_act1(self.dec_conv1(x))
        x = self.cbam7(x)

        x = self.dec_act2(self.dec_conv2(x))
        x = self.cbam8(x)
        x = self.upsample1(x)

        x = self.dec_act3(self.dec_conv3(x))
        x = self.cbam9(x)
        x = self.upsample2(x)

        x = self.dec_act4(self.dec_conv4(x))
        x = self.cbam10(x)
        x = self.upsample3(x)

        x = self.dec_act5(self.dec_conv5(x))
        x = self.cbam11(x)
        x = self.upsample4(x)

        x = self.dec_act6(self.dec_conv6(x))
        x = self.cbam12(x)

        # final output layer
        x = self.dec_conv7(x)
        return x
    
    
class CBAMDecoder(nn.Module):
    def __init__(self, activation_fn=nn.ReLU):
        super(CBAMDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(512),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(256),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(128),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(64),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(32),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(16),

            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # Final prediction
        )

    def forward(self, x):
        return self.dec(x)
    
######################################
#       MULTI-HEAD MODELS            #
######################################

class MultiHeadCBAMAutoencoder(nn.Module):
    def __init__(self, activation_fn=nn.ReLU):
        super(MultiHeadCBAMAutoencoder, self).__init__()

        # Shared Encoder with CBAM
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_RILEVATIONS_INPUT, 16, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(16),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(256),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            activation_fn(),
            CBAM(512),
            nn.MaxPool2d(2, 2),
        )

        # 4 decoder heads with CBAM
        self.decoders = nn.ModuleList([CBAMDecoder(activation_fn) for _ in range(4)])

    def forward(self, x):
        latent = self.encoder(x)
        return [decoder(latent) for decoder in self.decoders]