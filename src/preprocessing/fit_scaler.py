"""
Dataset Scaler Fitting Module

This module fits normalization scalers on the entire dataset using a chunked processing approach.
It processes large datasets that don't fit in memory by loading data in chunks, fitting
MinMaxScaler incrementally, and saving the trained scalers for later use during training
and inference.

Key features:
- Incremental fitting using sklearn's partial_fit for memory efficiency
- Chunk-based processing to handle large datasets
- Multi-sensor data processing with thread-based parallelization
- Deterministic data shuffling with fixed random seed
- Scaler persistence for consistent normalization across pipeline

The fitting process:
1. Divide dataset into manageable chunks
2. Process multiple sensors in parallel using ThreadPoolExecutor
3. Load and combine grid maps for each chunk
4. Incrementally fit MinMaxScaler on each chunk
5. Save fitted scalers for later use
"""

# Standard library imports
import sys
import os
import math
import pickle
import random
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants and functions
from config import *

# Generate each pair of grid map and bounding box map to be used for the partial fit
def generate_combined_grid_maps_fit(grid_map_path, grid_map_files, complete_grid_maps):
    
    for file in grid_map_files:
        complete_path = os.path.join(grid_map_path, file)

        points = load_points_grid_map(complete_path)
            
        grid_map_recreate = np.full((DIMENSION_Y, DIMENSION_X), NO_DATA_VALUE, dtype=float) # type: ignore

        cols, rows, heights = points.T
        grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

        complete_grid_maps.append(grid_map_recreate)

def process_lidar_chunk(lidar_directory, files_lidar_chunck, complete_grid_maps):
    """
    Process a chunk of LiDAR files for a single sensor.
    
    This function loads and processes a subset of LiDAR grid files for one sensor,
    adding the resulting grid maps to the complete collection. It serves as a
    wrapper around the grid map generation function to enable parallel processing.
    
    Args:
        lidar_directory (str): Path to the directory containing LiDAR grid files
        files_lidar_chunck (list): List of filenames to process in this chunk
        complete_grid_maps (list): List to store the processed grid maps
    
    Returns:
        list: Updated complete_grid_maps list with new grid maps added
    """
    # Generate combined grid maps for this chunk and add to the complete collection
    generate_combined_grid_maps_fit(lidar_directory, files_lidar_chunck, complete_grid_maps) # type: ignore

    return complete_grid_maps

def fit_scalers(lidar_paths):
    """
    Fit MinMaxScaler on the entire dataset using chunked processing.
    
    This function implements an incremental learning approach to fit normalization
    scalers on large datasets that don't fit in memory. It processes data in chunks,
    with each chunk containing samples from all sensors, and uses sklearn's partial_fit
    to incrementally update the scaler parameters.
    
    Args:
        lidar_paths (list): List of directory paths for each LiDAR sensor
    
    Returns:
        None: Saves fitted scalers to disk
        
    Process:
        1. Initialize MinMaxScaler for input data normalization
        2. Prepare file lists for each sensor with deterministic shuffling
        3. Calculate chunk sizes based on total files and number of chunks
        4. For each chunk:
           a. Load data from all sensors in parallel
           b. Combine and shuffle the data
           c. Incrementally fit the scaler using partial_fit
        5. Save fitted scalers to disk for later use
    """
    # Initialize the MinMaxScaler for input data normalization
    scaler_X = MinMaxScaler()

    # Get number of chunks from configuration
    number_of_chucks = NUMBER_OF_CHUNCKS_TRAIN

    # Set random seed for reproducible shuffling
    random.seed(SEED)

    # Initialize lists to store file information for each sensor
    files_lidar = []        # List of file lists, one per sensor
    files_for_chunck = []   # Number of files per chunk for each sensor

    # Prepare file lists for each sensor
    for i in range(NUMBER_OF_SENSORS):
        # Get sorted list of files for this sensor and shuffle deterministically
        files_lidar_1 = list((sorted([f for f in os.listdir(lidar_paths[i])])))
        random.shuffle(files_lidar_1)
        
        # Store the shuffled file list for this sensor
        files_lidar.append(list(files_lidar_1))

    # Calculate chunk sizes for each sensor
    for i in range(NUMBER_OF_SENSORS):
        # Calculate files per chunk (rounded up to ensure all files are included)
        files_for_chunck.append(math.ceil(len(files_lidar[i]) / number_of_chucks)) #type: ignore

    # Process each chunk sequentially for incremental fitting
    for i in range(number_of_chucks): #type: ignore
        
        # Initialize list to collect grid maps from all sensors for this chunk
        complete_grid_maps = []

        # Prepare file chunks for each sensor
        all_files_chunck = []
        for k in range(NUMBER_OF_SENSORS):
            # Calculate start and end indices for this chunk
            start_idx = i * files_for_chunck[k]
            end_idx = min((i+1) * files_for_chunck[k], len(files_lidar[k]))
            
            # Extract files for this chunk from this sensor
            files_lidar_chunck = files_lidar[k][start_idx:end_idx] #type: ignore
            all_files_chunck.append(files_lidar_chunck)

        # Process all sensors in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUMBER_OF_SENSORS) as executor:
            futures = []
            
            # Submit processing tasks for each sensor
            for j in range(NUMBER_OF_SENSORS):
                futures.append(executor.submit(process_lidar_chunk, lidar_paths[j], all_files_chunck[j], complete_grid_maps))
            
            # Collect results from all sensor processing tasks
            for future in futures:
                complete_grid_maps = future.result()
        
        # Shuffle the combined data to ensure random mixing of sensor data
        random.shuffle(complete_grid_maps)

        # Convert list to numpy array for scaler fitting
        complete_grid_maps = np.array(complete_grid_maps)

        # Incrementally fit the scaler on this chunk of data
        scaler_X.partial_fit(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1]))

    # Save the fitted scalers to disk for later use
    save_directory = SCALER_DIR

    # Ensure the output directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save the fitted scaler using pickle for persistence
    with open(os.path.join(save_directory, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    print("scaler saved")


# Main execution block
if __name__ == "__main__":
    
    # Build list of LiDAR grid directories for all sensors
    lidar_direcory_list = []
    
    # Create directory path for each sensor by replacing 'X' with sensor number
    for i in range(1, NUMBER_OF_SENSORS+1):
        lidar_direcory_list.append(HEIGHTMAP_DIRECTORY.replace('X', str(i)))

    # Fit scalers on the entire multi-sensor dataset
    fit_scalers(lidar_direcory_list)
