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

# Generate each pair of grid map and bounding box map to be used for the partial fit
def generate_combined_grid_maps_fit(grid_map_path, grid_map_BB_path, grid_map_files, grid_map_BB_files, complete_grid_maps, complete_grid_maps_BB, complete_num_BB, bool_value):
    for file, file_BB in zip(grid_map_files, grid_map_BB_files):
        complete_path = os.path.join(grid_map_path, file)
        complete_path_BB = os.path.join(grid_map_BB_path, file_BB)
        #print(f"Loading {file} and {file_BB}...")

        points = load_points_grid_map(complete_path)
        points_BB, indeces = load_points_grid_map_BB(complete_path_BB)

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

        actors = {}
        
        grid_map_group = []

        for j in range (NUMBER_RILEVATIONS_INPUT):

            complete_path = os.path.join(grid_map_path, grid_map_files[i][j])

            points = load_points_grid_map(complete_path)

            grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float) # type: ignore

            cols, rows, heights = points.T
            grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

            grid_map_group.append(grid_map_recreate)

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

        #Save the input
        for k in range(len(FUTURE_TARGET_RILEVATION)):   

            complete_path_BB = os.path.join(grid_map_BB_path, grid_map_files[i][-len(FUTURE_TARGET_RILEVATION) + k])

            points_BB, indeces = load_points_grid_map_BB(complete_path_BB)

            all_pairs = []

            # Iterate through each row in the numpy array
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
    for i in range(0, len(files_lidar) - NUMBER_RILEVATIONS_INPUT - FUTURE_TARGET_RILEVATION[-1] + 1, NUMBER_RILEVATIONS_INPUT):
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
    #print("len combined files", len(combined_list))
    #print("len conbined list:", len(combined_list[2]))
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

        #print("grid_map shape", grid_map.shape)
        #print("grid_map_BB shape", grid_map_BB.shape)

        # Ensure grid_map_BB has the same shape as grid_map
        #grid_map_BB = np.squeeze(grid_map_BB)  # Remove the extra dimension (if present)

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

        #grid_map_BB = np.expand_dims(grid_map_BB, axis=0)  # Add the extra dimension back
        
        #print("first and second augmentation", first_augmentation, second_augmentation)

        augmented_grid_maps.append(grid_map)
        augmented_grid_maps_BB.append(grid_map_BB)

        #print("first augmentation", first_augmentation)
        #print("second augmentation", second_augmentation)

        '''
        fig, ax = plt.subplots(4, 2, figsize=(10, 10))
        ax[0,0].imshow(random_gm[i][0], cmap='gray')
        ax[0,1].imshow(random_BB[i][0], cmap='gray')

        ax[1,0].imshow(grid_map[0], cmap='gray')
        ax[1,1].imshow(grid_map_BB[0], cmap='gray')

        ax[2,0].imshow(random_gm[i][3], cmap='gray')
        ax[2,1].imshow(random_BB[i][3], cmap='gray')

        ax[3,0].imshow(grid_map[3], cmap='gray')
        ax[3,1].imshow(grid_map_BB[3], cmap='gray')

        plt.show()
        '''

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

def check_dead_neurons_autoencoder(model, input_data, target_data, activation_fn=nn.ReLU):
    model.eval()
    dead_neurons = {}
    
    def hook_fn(module, input, output, layer_name):
        # Debugging: Print layer name and output shape
        #print(f"Hook triggered for layer: {layer_name}, output shape: {output.shape}")
        
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

def check_dead_neurons(model, input_data, target_data, activation_fn=nn.ReLU):
    model.eval()
    dead_neurons = {}
    
    def hook_fn(module, input, output, layer_name):
        # Debugging: Print layer name and output shape
        #print(f"Hook triggered for layer: {layer_name}, output shape: {output.shape}")
        
        num_zeros = (output == 0).sum().item()
        total_neurons = output.numel()
        zero_percentage = (num_zeros / total_neurons) * 100
        dead_neurons[layer_name] = zero_percentage

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, activation_fn):  
            hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))
    
    alpha_cumprod = get_noise_schedule()
    t = torch.randint(5, RANGE_TIMESTEPS, (input_data.shape[0],))  # Random timestep
    target_data = target_data.float()
    noisy_target, noise = get_noisy_target(target_data, alpha_cumprod, t)
    t_tensor = t.view(-1, 1, 1, 1).expand_as(target_data)  # Reshape and expand to match targets' shape
    t_tensor = t_tensor / (RANGE_TIMESTEPS - 1)  # Normalize t_tensor to scale values between 0 and 1                   
    t_tensor = t_tensor.to(target_data.device)  # Move t_tensor to GPU

    with torch.no_grad():
        _ = model(input_data, noisy_target, t_tensor)  # Forward pass to collect activations

    for hook in hooks:
        hook.remove()  # Clean up hooks

    print("\nDead Neurons:")
    for layer, percentage in dead_neurons.items():
        print(f"Layer {layer}: {percentage:.2f}% dead neurons")
    print("\n")

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(1, dim)  # Map scalar timestep to embedding of size 'dim'

    def forward(self, timesteps):
        # Now timesteps is expected to be of shape (batch_size, 1)
        timesteps = timesteps.float()  # Ensure the type is float
        print(timesteps)
        print(timesteps.shape)
        print(self.linear(timesteps).shape)
        return self.linear(timesteps)   # Returns tensor of shape (batch_size, dim)

# Dummy noise scheduler for illustration (replace with your actual DDPMScheduler)
class DDPMScheduler:
    def __init__(self, num_train_timesteps, beta_schedule):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule

    def add_noise(self, x, noise, timestep):
        # A simple noise addition for demonstration
        # In practice, the noise addition is more complex and depends on timestep
        return x + noise

# Revised DiffusionPredictor class
class DiffusionPredictor(nn.Module):
    def __init__(self, img_size=400, in_channels=5, out_channels=1, time_embed_dim=128):
        super().__init__()

        # Initialize UNet (configure with your actual parameters)
        self.unet = UNet2DModel(
            sample_size=img_size,
            in_channels=in_channels,  # e.g., 5 past frames as input
            out_channels=out_channels,  # e.g., predict 1 future frame
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

        # Timestep embedding
        self.time_embedding = TimestepEmbedding(time_embed_dim)
        
        # Noise scheduler (DDPM for basic diffusion training)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    def forward(self, x):
        """
        Forward pass for general diffusion training.
        Expected x shape: (batch_size, 5, 400, 400)
        """
        batch_size = x.size(0)
        
        # Instead of using a fixed timestep, we sample a random timestep for each sample.
        # This samples an integer from [0, num_train_timesteps)
        timestep = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size, 1), device=x.device)
        
        # Create the timestep embedding (shape: (batch_size, time_embed_dim))
        timestep_emb = self.time_embedding(timestep)
        
        # Add noise to the input image x
        noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.add_noise(x, noise, timestep)
        
        # Use UNet to predict the noise at the current timestep.
        noise_pred = self.unet(noisy_x, timestep_emb)
        return noise_pred

class Autoencoder_classic(nn.Module):
    def __init__(self, activation_fn=nn.ReLU): # Constructor method for the autoencoder
        super(Autoencoder_classic, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_RILEVATIONS_INPUT, 32, kernel_size=3, padding=1),
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

class Autoencoder_big_big(nn.Module):
    def __init__(self, activation_fn=nn.ReLU): # Constructor method for the autoencoder
        super(Autoencoder_big_big, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(NUMBER_RILEVATIONS_INPUT, 16, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride = 2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
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
    #print("min and max values of noise", noise.min(), noise.max())
    #print("min and max values of x0", x0.min(), x0.max())
    noise = noise.to(x0.device)

    # Gather alpha_cumprod[t] for each sample in the batch
    alpha_t = alpha_cumprod[t].view(-1, 1, 1, 1)  # Reshape for broadcasting to have shape (B, C, H, W)
    alpha_t = alpha_t.to(x0.device)

    # Apply the forward diffusion equation
    x_t = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    #print("min and max values before norm", x_t.min(), x_t.max())

    #print("x_t shape before norm", x_t.shape)

    # Normalize x_t to the range [0, 1]
    x_t = torch.clamp(x_t, 0, 1)  # Ensure values are in the range [0, 1]
    #print("min and max values", x_t.min(), x_t.max())

    #print("x_t shape", x_t.shape)

    return x_t, noise  # Return both x_t and the added noise for training

class ConditionalUNet(nn.Module):
    def __init__(self, input_channels=7, hidden_dim=64, activation_fn=nn.ReLU):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            activation_fn(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            activation_fn()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            activation_fn(),
            nn.Conv2d(hidden_dim, 1, 3, padding=1)
        )

    def forward(self, x_t, past_frames, timestep):
        x = torch.cat([x_t, past_frames, timestep], dim=1)  # Concatenate past frames with noisy target
        h = self.encoder(x)
        out = self.decoder(h)
        return out  # Predict noise (ε)

class BigUNet(nn.Module):
    def __init__(self, input_channels=7, output_channels=1, features=[16, 32, 64, 128], activation_fn=nn.ReLU):
        super(BigUNet, self).__init__()
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

class BigUNet_autoencoder(nn.Module):
    def __init__(self, input_channels=NUMBER_RILEVATIONS_INPUT, output_channels=1, features=[16, 32, 64, 128], activation_fn=nn.ReLU):
        super(BigUNet_autoencoder, self).__init__()
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_step(model, optimizer, past_frames, future_frame, alpha_cumprod, T):
    device = past_frames.device  # Ensure everything is on the same device
    t = torch.randint(5, T, (past_frames.shape[0],))  # Random timestep for each sample
    noise = torch.randn_like(future_frame, device=device)  # Sample random noise

    # Start with a noisy version of the future frame
    sqrt_alpha_cumprod = alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
    noisy_future = sqrt_alpha_cumprod * future_frame + (1 - sqrt_alpha_cumprod) * noise

    # Now we run for t steps (not T steps, only until the selected timestep)
    total_loss = 0
    torch.cuda.empty_cache()
    for batch_idx in range(past_frames.shape[0]):  # Iterate over each sample in the batch
        
        noisy_future = sqrt_alpha_cumprod[batch_idx] * future_frame[batch_idx] + (1 - sqrt_alpha_cumprod[batch_idx]) * noise
        
        for step in range(t[batch_idx], -1, -1):  # Iterate backwards from selected t to 0
            # Predict the noise for this timestep
            predicted_noise = model(noisy_future, past_frames)

            # Update the noisy frame by removing predicted noise (refine frame)
            noisy_future = noisy_future - predicted_noise  # Refine frame with denoising

            # Calculate loss with true noise
            loss = F.mse_loss(predicted_noise, noise)
            total_loss += loss
            torch.cuda.empty_cache()

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

from tqdm import tqdm

def train_model(model, dataloader, epochs=10, T=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move model to GPU (if available)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    summary(model, input_size=[(5, 400, 400), (1, 400, 400), (1,400,400)])
    # Get noise schedule
    alpha_cumprod = get_noise_schedule(T)
    print(alpha_cumprod)

    torch.cuda.empty_cache()

    for epoch in range(epochs):
        total_loss = 0  # Track total loss for the epoch
        for past_frames, future_frame in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            past_frames, future_frame = past_frames.to(device), future_frame.to(device)  # Move to GPU if needed

            # Call `train_step` for each pair
            loss = train_step(model, optimizer, past_frames, future_frame, alpha_cumprod, T)
            total_loss += loss  # Accumulate loss
        
        torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)  # Compute average loss
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")  # Print epoch loss

def sample_future_frame(model, past_frames, T=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    past_frames = past_frames.to(device)

    x_t = torch.randn_like(past_frames[:, :1, :, :])  # Start from pure noise

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        predicted_noise = model(x_t, past_frames, t_tensor)
        x_t = x_t - predicted_noise  # Denoising step

    return x_t  # Predicted future frame

def train_step_old(model, optimizer, past_frames, future_frame, alpha_cumprod, T):
    device = past_frames.device  # Ensure everything is on the same device
    t = torch.randint(0, T, (past_frames.shape[0],), device=device)  # Random timestep
    noise = torch.randn_like(future_frame, device=device)  # Sample random noise

    # Start with a noisy version of the future frame
    sqrt_alpha_cumprod = alpha_cumprod.to(device)[t].view(-1, 1, 1, 1)
    noisy_future = sqrt_alpha_cumprod * future_frame + (1 - sqrt_alpha_cumprod) * noise

    # Now we run for T steps
    total_loss = 0
    for step in range(T-1, -1, -1):  # Iterating backwards from T-1 to 0
        t_tensor = torch.tensor([step], dtype=torch.float32, device=device)  # Current timestep

        # Predict the noise for this timestep
        predicted_noise = model(noisy_future, past_frames)

        # Update the noisy frame by removing predicted noise
        noisy_future = noisy_future - predicted_noise  # Refine frame with denoising

        # Calculate loss with true noise
        loss = F.mse_loss(predicted_noise, noise)
        total_loss += loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.item()

def calculate_dead_neuron(model, device):
    # Function to calculate dead neuron percentage
    def dead_neuron_percentage(activations):
        # activations: (batch_size, num_neurons, height, width)
        print(activations.shape)
        num_neurons = activations.shape[1] * activations.shape[2] * activations.shape[3]
        # For each neuron, check if it was always zero across the batch
        dead_neurons = (activations == 0).all(dim=(0, 2, 3)).sum().item()
        return 100.0 * dead_neurons / activations.shape[1]
    
    # Get the first batch of data
    loader = load_dataset('val', 0, device, 16)
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