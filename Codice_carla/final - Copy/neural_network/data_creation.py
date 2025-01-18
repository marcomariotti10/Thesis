import tensorflow as tf
import csv
import os
import numpy as np
from tensorflow import keras as tfk
from keras import layers as tfkl
import sys
import sklearn
from sklearn.model_selection import train_test_split

# Import constants and catch any import errors
from link_to_constants import *
link_constants()
try:
    from constants import * # type: ignore
    print("Successfully imported constants.")
except ImportError as e:
    print(f"Error importing constants: {e}")


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
    print(np_points)
    return np_points

def generate_grid_map (grid_map_path):
    grid_map_files = [f for f in os.listdir(grid_map_path) if f.endswith('.cvs')]

    list_grid_maps = []

    for file in grid_map_files:
        print(file)
        complete_path = os.path.join(grid_map_path, file)
        print(f"Loading {file}...")
        points = load_points_grid_map(complete_path)

        # Recreate the grid map from positions array
        grid_map_recreate = np.full((Y_RANGE, X_RANGE), MIN_HEIGHT) # type: ignore

        # Fill the grid map with values from positions array
        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = height

        list_grid_maps.append(grid_map_recreate)
    
    return list_grid_maps

def load_points_grid_map_BB (csv_file):
    """Load bounding box vertices from a CSV file."""
    points = []
    heights = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            coordinates = [ [ float(row[i]), float(row[i+1]) ] for i in range(2, 10, 2)]
            height = (float(row[10]))
            points.append(coordinates)
            heights.append(height)
    np_points = np.array(points)
    np_heights = np.array(heights)
    return np_points, np_heights

def generate_grid_map_BB (grid_map_path):
    grid_map_files = [f for f in os.listdir(grid_map_path) if f.endswith('.cvs')]

    list_grid_maps = []

    for file in grid_map_files:
        print(file)
        grid_map_path = os.path.join(grid_map_path, file)
        print(f"Loading {file}...")
        points, heights = load_points_grid_map_BB(grid_map_path)

        # Recreate the grid map from positions array
        grid_map_recreate = np.full((Y_RANGE, X_RANGE), MIN_HEIGHT) # type: ignore

        # Fill the grid map with values from positions array
        for pos,i in enumerate(points):
            col, row = pos
            grid_map_recreate[int(row), int(col)] = heights[i]
    
    return list_grid_maps

def split_data(lidar_data, BB_data):
    # Split the dataset into a combined training and validation set, and a separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        lidar_data, # Samples
        BB_data, # Labels
        test_size = 0.1,
        random_state=SEED
    )
    return X_train_val, X_test, y_train_val, y_test