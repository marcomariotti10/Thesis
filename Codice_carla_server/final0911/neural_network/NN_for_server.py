import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
from matplotlib.path import Path
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import MinMaxScaler

def import_constants():
    # Dynamically construct the path to the data_gen_and_processing folder
    current_dir = os.getcwd()
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

def generate_combined_grid_maps(grid_map_path, grid_map_BB_path, complete_grid_maps, complete_grid_maps_BB, complete_num_BB):
    grid_map_files = sorted([f for f in os.listdir(grid_map_path)])
    grid_map_BB_files = sorted([f for f in os.listdir(grid_map_BB_path)])
    
    for file, file_BB in zip(grid_map_files, grid_map_BB_files):
        # Import constants inside the function
        from constants import Y_RANGE, X_RANGE, FLOOR_HEIGHT

        complete_path = os.path.join(grid_map_path, file)
        complete_path_BB = os.path.join(grid_map_BB_path, file_BB)
        print(f"Loading {file} and {file_BB}...")

        points_BB = load_points_grid_map_BB(complete_path_BB)

        num_BB = points_BB.shape[0]
        if num_BB == 0:
            continue
        
        points = load_points_grid_map(complete_path)

        grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=int)
        grid_map_recreate_BB = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=int)

        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = int(height)

        for i in range(len(points_BB)):
            vertices = np.array(points_BB[i])
            height_BB = int(vertices[0, 2])  # Assuming all vertices have the same height
            fill_polygon(grid_map_recreate_BB, vertices, height_BB)

        complete_grid_maps.append(grid_map_recreate)
        complete_grid_maps_BB.append(grid_map_recreate_BB)
        complete_num_BB.append(num_BB)

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

def visualize_prediction(prediction, real):
    """
    Visualize the grid map and the prediction.
    
    Parameters:
    - grid_map: numpy array of shape (400, 400)
    - prediction: numpy array of shape (400, 400)
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(prediction, cmap='gray')
    ax[0].set_title('Prediction Map')
    
    ax[1].imshow(real, cmap='gray')
    ax[1].set_title('Real Map')
    
    plt.show()

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

class WeightedCustomLoss(nn.Module):
    def __init__(self, weight=100):
        super(WeightedCustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.weight = weight

    def forward(self, predictions, targets):
        from constants import FLOOR_HEIGHT
        mask = (targets != FLOOR_HEIGHT).float()
        masked_predictions = predictions * mask
        masked_targets = targets * mask
        loss = self.mse_loss(masked_predictions, masked_targets)
        
        # Apply weighting to the loss
        weighted_loss = loss * self.weight + self.mse_loss(predictions * (1 - mask), targets * (1 - mask))
        return weighted_loss

class LargeDataset(Dataset):
    def __init__(self, grid_map_path, grid_map_BB_path, transform=None):
        self.grid_map_path = grid_map_path
        self.grid_map_BB_path = grid_map_BB_path
        self.grid_map_files = sorted([f for f in os.listdir(grid_map_path)])
        self.grid_map_BB_files = sorted([f for f in os.listdir(grid_map_BB_path)])
        self.transform = transform

    def __len__(self):
        return len(self.grid_map_files)

    def __getitem__(self, idx):
        grid_map_file = self.grid_map_files[idx]
        grid_map_BB_file = self.grid_map_BB_files[idx]

        complete_path = os.path.join(self.grid_map_path, grid_map_file)
        complete_path_BB = os.path.join(self.grid_map_BB_path, grid_map_BB_file)

        points_BB = load_points_grid_map_BB(complete_path_BB)
        points = load_points_grid_map(complete_path)

        grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=int)
        grid_map_recreate_BB = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=int)

        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = int(height)

        for i in range(len(points_BB)):
            vertices = np.array(points_BB[i])
            height_BB = int(vertices[0, 2])  # Assuming all vertices have the same height
            fill_polygon(grid_map_recreate_BB, vertices, height_BB)

        if self.transform:
            grid_map_recreate = self.transform(grid_map_recreate)
            grid_map_recreate_BB = self.transform(grid_map_recreate_BB)

        return grid_map_recreate, grid_map_recreate_BB

if __name__ == "__main__":
    
    gc.collect()
    
    import_constants()
    
    try:
        from constants import *  # type: ignore
        print("Successfully imported constants.")
    except ImportError as e:
        print(f"Error importing constants: {e}")

    set_start_method("spawn", force=True)

    model = Autoencoder()
    model.apply(weights_init)
    criterion = WeightedCustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    summary(model, (1, 400, 400))

    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        torch.cuda.empty_cache()

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')
    torch.cuda.empty_cache()

    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions).cpu().numpy()
    print("Predictions Shape:", predictions.shape)
    torch.cuda.empty_cache()