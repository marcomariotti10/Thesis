import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torchvision
import csv
import numpy as np
from torchsummary import summary
import torch.nn.init as init
import random
import cv2
import ast
import platform

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice
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

def load_array(file_path):
    return np.load(file_path)

def load_dataset(name,i,device, batch):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_name_ffcv_path = os.path.join(FFCV_DIRECTORY)
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

############################################
#       NEURAL NETWORK FUNCTIONS           #
############################################

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

# Linear noise schedule function
def get_noise_schedule():
    beta_t = torch.linspace(MINIMUM_BETHA, MAXIMUM_BETHA, RANGE_TIMESTEPS)  # Noise schedule
    alpha_t = 1.0 - beta_t
    alpha_cumprod = torch.cumprod(alpha_t, dim=0)  # Cumulative product of alpha
    return beta_t, alpha_t, alpha_cumprod

def get_noisy_target(x0, alpha_cumprod, t):
    """
    Adds noise to the future target (x0) based on the diffusion process.
    
    Args:
        x0 (torch.Tensor): The ground truth future binary grid map (B, 1, 400, 400).
        alpha_cumprod (torch.Tensor): Precomputed cumulative product of alpha values.
        t (torch.Tensor): Timestep indices (B,).
    
    Returns:
        x_t (torch.Tensor): The noisy future frame at timestep t.
        noise (torch.Tensor): The added Gaussian noise.
    """

    # Sample Gaussian noise with the same shape as x0
    noise = torch.randn_like(x0)

    noise = noise.to(x0.device)

    # Gather alpha_cumprod[t] for each sample in the batch
    alpha_cumprod = alpha_cumprod.to(x0.device)
    alpha_t = alpha_cumprod[t].view(-1, 1, 1, 1)  # Reshape for broadcasting to have shape (B, C, H, W)
    alpha_t = alpha_t.to(x0.device)

    # Apply the forward diffusion equation
    x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    return x_t, noise  # Return both x_t and the added noise for training