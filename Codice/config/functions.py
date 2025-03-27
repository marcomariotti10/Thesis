import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
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
            coordinates = [row[2]]
            points.append(coordinates)

    np_points = np.array(points)
    return np_points

# Generate each pair of grid map and bounding box map to be used for the partial fit
def generate_combined_grid_maps_fit(grid_map_path, grid_map_BB_path, grid_map_files, grid_map_BB_files, complete_grid_maps, complete_grid_maps_BB, complete_num_BB, bool_value):
    for file, file_BB in zip(grid_map_files, grid_map_BB_files):
        complete_path = os.path.join(grid_map_path, file)
        complete_path_BB = os.path.join(grid_map_BB_path, file_BB)
        #print(f"Loading {file} and {file_BB}...")

        points = load_points_grid_map(complete_path)
        points_BB = load_points_grid_map_BB(complete_path_BB)

        all_pairs = []

        # Iterate through each row in the numpy array
        for row in points_BB:
            # Extract the string from the array
            string_data = row[0]
            # Safely evaluate the string to convert it into a list of tuples
            pairs = ast.literal_eval(string_data)
            # Add the pairs to the all_pairs list
            all_pairs.extend(pairs)
            
        all_pairs = np.array(all_pairs)

        grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float) # type: ignore
        grid_map_recreate_BB = np.full((Y_RANGE, X_RANGE), 0, dtype=float) # type: ignore

        cols, rows, heights = points.T
        grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

        if len(all_pairs) != 0:
            cols, rows = all_pairs.T
            grid_map_recreate_BB[rows.astype(int), cols.astype(int)] = 1
        
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

def generate_combined_grid_maps_pred(grid_map_path, grid_map_BB_path, grid_map_files, complete_grid_maps, complete_grid_maps_BB):
    
    #print(len(grid_map_files))
    
    for i in range(len(grid_map_files)):
        
        grid_map_group = []

        for j in range (NUMBER_RILEVATIONS_INPUT):
            complete_path = os.path.join(grid_map_path, grid_map_files[i][j])

            points = load_points_grid_map(complete_path)

            grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float) # type: ignore

            cols, rows, heights = points.T
            grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

            grid_map_group.append(grid_map_recreate)
        
        #Save the input   
        complete_grid_maps.append(grid_map_group)

        complete_path_BB = os.path.join(grid_map_BB_path, grid_map_files[i][-1])

        points_BB = load_points_grid_map_BB(complete_path_BB)

        all_pairs = []

        # Iterate through each row in the numpy array
        for row in points_BB:
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

        complete_grid_maps_BB.append(grid_map_recreate_BB)

def generate_combined_list(files_lidar, files_BB):
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
    #print("\nlen files_lidar:", len(files_lidar))
    for i in range(0, len(files_lidar) - NUMBER_RILEVATIONS_INPUT - FUTURE_TARGET_RILEVATION + 1, NUMBER_RILEVATIONS_INPUT):
        # Take NUMBER_RILEVATIONS_INPUT lidar files starting from the current index
        lidar_group = files_lidar[i:i + NUMBER_RILEVATIONS_INPUT]
        # Take the FUTURE_TARGET_RILEVATION-th BB file after the last lidar file
        BB_file = files_BB[i + NUMBER_RILEVATIONS_INPUT + FUTURE_TARGET_RILEVATION - 1]
        # Combine the lidar group and the BB file into one element
        combined_list.append(lidar_group + [BB_file])

    random.shuffle(combined_list)
    #print("len combined files", len(combined_list))
    return combined_list

def fill_polygon(grid_map, vertices, height):
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

        # Ensure grid_map_BB has the same shape as grid_map
        grid_map_BB = np.squeeze(grid_map_BB)  # Remove the extra dimension (if present)

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
                augmentation = lambda img: random_shift(img, axis=axis, shift=shift)
            elif augmentation_type[0] == 'flip':
                flip_code = random.choice([0, 1])  # 0 for vertical flip, 1 for horizontal flip
                augmentation = lambda img: cv2.flip(img, flip_code)

            # Apply the augmentation to all images in the group
            for k in range(grid_map.shape[0]):
                grid_map[k] = augmentation(grid_map[k])
            grid_map_BB = augmentation(grid_map_BB)

        grid_map_BB = np.expand_dims(grid_map_BB, axis=0)  # Add the extra dimension back
        
        #print("first and second augmentation", first_augmentation, second_augmentation)

        augmented_grid_maps.append(grid_map)
        augmented_grid_maps_BB.append(grid_map_BB)
    
    return augmented_grid_maps, augmented_grid_maps_BB


def rotate_image(image, angle):
    """
    Rotate an image by a specified angle.
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
    return np.load(file_path)

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

############################################
#       NEURAL NETWORK FUNCTIONS           #
############################################

def load_dataset(name,i,device, batch):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
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


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def check_dead_neurons(model, input_data, activation_fn=nn.ReLU):
    model.eval()
    dead_neurons = {}
    
    def hook_fn(module, input, output, layer_name):
        num_zeros = (output == 0).sum().item()
        total_neurons = output.numel()
        zero_percentage = (num_zeros / total_neurons) * 100
        dead_neurons[layer_name] = zero_percentage

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, activation_fn):  
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

    with torch.no_grad():
        _ = model(input_data)  # Forward pass to collect activations

    for hook in hooks:
        hook.remove()  # Clean up hooks

    print("\nDead Neurons:")

    for layer, percentage in dead_neurons.items():
        print(f"Layer {layer}: {percentage:.2f}% dead neurons")

    print("\n")

class Autoencoder_classic(nn.Module):
    def __init__(self, activation_fn=nn.ReLU): # Constructor method for the autoencoder
        super(Autoencoder_classic, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
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

class Autoencoder_big(nn.Module):
    def __init__(self, activation_fn=nn.ReLU): # Constructor method for the autoencoder
        super(Autoencoder_big, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
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
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128], activation_fn=nn.ReLU):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder path
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature, activation_fn ))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, activation_fn)
        
        # Decoder path
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv(feature * 2, feature, activation_fn))
        
        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
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
            x = torch.cat((skip_connection, x), dim=1)  # Concatenate skip connection
            x = self.decoder[i + 1](x)  # DoubleConv layer
    
        return self.final_conv(x)  # Sigmoid for binary segmentation

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