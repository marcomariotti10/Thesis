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

if __name__ == "__main__":
    
    gc.collect()

    import_constants()
    try:
        from constants import * # type: ignore
        print("Successfully imported constants.")
    except ImportError as e:
        print(f"Error importing constants: {e}")
    # Ensure the start method is set as early as possible

    set_start_method("spawn", force=True)

    complete_grid_maps = []
    complete_grid_maps_BB = []
    complete_num_BB = []

    # Load sensor1
    generate_combined_grid_maps(LIDAR_1_GRID_DIRECTORY, NEW_POSITIONS_LIDAR_1_GRID_DIRECTORY, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

    # Load sensor2
    generate_combined_grid_maps(LIDAR_2_GRID_DIRECTORY, NEW_POSITIONS_LIDAR_2_GRID_DIRECTORY, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

    # Load sensor3
    generate_combined_grid_maps(LIDAR_3_GRID_DIRECTORY, NEW_POSITIONS_LIDAR_3_GRID_DIRECTORY, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

    #print(f"Number of grid maps for each sensor: {len(complete_grid_maps[0])}, {len(complete_grid_maps[1])}, {len(complete_grid_maps[2])}")
    #print(f"Number of grid maps BB for each sensor: {len(complete_grid_maps_BB[0])}, {len(complete_grid_maps_BB[1])}, {len(complete_grid_maps_BB[2])}")
    #print(f"Number of bounding boxes for each sensor: {len(complete_num_BB[0])}, {len(complete_num_BB[1])}, {len(complete_num_BB[2])}")

    # Concatenate the lists in complete_grid_maps along the first dimension
    complete_grid_maps = np.array(complete_grid_maps)
    print(f"complete grid map shape : {complete_grid_maps.shape}")

    # Concatenate the lists in complete_grid_maps_BB along the first dimension
    complete_grid_maps_BB = np.array(complete_grid_maps_BB)
    print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

    complete_num_BB = np.array(complete_num_BB)
    print(f"complete number of bounding boxes shape : {complete_num_BB.shape}")

    complete_num_BB = np.expand_dims(complete_num_BB, axis=1)
    print(f"expanded number of bounding boxes shape : {complete_num_BB.shape}")

    # ELiminate all the data points with 0 bounding boxes
    total_num_greater_than_1 = 0
    total_num_equal_to_0 = 0
    for i in range(complete_num_BB.shape[0]):
        if (complete_num_BB[i] >= 1):
            total_num_greater_than_1 += 1
        else:
            total_num_equal_to_0 += 1

    print(total_num_greater_than_1, total_num_equal_to_0)

    # Split the data
    X_train_val, X_test, y_train_val, y_test, num_BB_train_val, num_BB_test = split_data(complete_grid_maps, complete_grid_maps_BB, complete_num_BB, TEST_SIZE) # type: ignore

    del complete_grid_maps, complete_grid_maps_BB, complete_num_BB

    X_train, X_val, y_train, y_val, num_BB_train, num_BB_val = split_data(X_train_val, y_train_val, num_BB_train_val, len(X_test)) # Esure that val and test set have the same lenght

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape, num_BB_train.shape, num_BB_val.shape, num_BB_test.shape)
    
    sum_train = 0
    sum_val = 0
    sum_test = 0

    for i in range(len(num_BB_train)):
        sum_train += num_BB_train[i]
    print(f"Sum_train: ", sum_train)
    print(f"Average_sum_train: ",sum_train/len(num_BB_train))

    for i in range(len(num_BB_val)):
        sum_val += num_BB_val[i]
    print(f"Sum_val: ",sum_val)
    print(f"Average_sum_val: ",sum_val/len(num_BB_val))

    for i in range(len(num_BB_test)):
        sum_test += num_BB_test[i]
    print(f"Sum_test: ",sum_test)
    print(f"Average_sum_test: ",sum_test/len(num_BB_test))

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Normalize the data
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    y_train = scaler.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_val = scaler.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
    y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    print("New input shape (train, val, test):", X_train.shape, X_val.shape, X_test.shape)

    y_train = np.expand_dims(y_train, axis=1)
    y_val = np.expand_dims(y_val, axis=1)
    y_test = np.expand_dims(y_test, axis=1)

    print("New labels shape (train, val, test):", y_train.shape, y_val.shape, y_test.shape)
        
    # Prepare data loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.
    val_dataset = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    del X_train, X_val, X_test, y_train, y_val, y_test

    print("Len dataset (train, val, test):",len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    del train_dataset, val_dataset, test_dataset

    model = Autoencoder()
    model.apply(weights_init)
    criterion = WeightedCustomLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01) # model.parameters() passes the parameters of the model to the optimizer so that it can update them during training.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    summary(model, (1, 400, 400))
    
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)

    num_epochs = 50
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)  # Average loss over all batches

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)  # Average loss over all batches

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Evaluate on test set
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
    # Clear GPU cache
    torch.cuda.empty_cache()


    # Make predictions
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            inputs, _ = data
            inputs= inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs)
    predictions = torch.cat(predictions).cpu().numpy()
    print("Predictions Shape:", predictions.shape)
    # Clear GPU cache
    torch.cuda.empty_cache()
