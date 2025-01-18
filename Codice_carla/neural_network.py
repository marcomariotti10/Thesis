import tensorflow as tf
import tensorflow as tf
import csv
import os
import numpy as np
from tensorflow import keras as tfk
from keras import layers as tfkl
from constants import *

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

def generate_grid_map (grid_map_path):
    grid_map_files = [f for f in os.listdir(grid_map_path) if f.endswith('.cvs')]

    for file in grid_map_files:
            print(file)
            grid_map_path = os.path.join(grid_map_path, file)
            print(f"Loading {file}...")
            points = load_points_grid_map(grid_map_path)
            print(points)

            min_height = MIN_HEIGHT
            x_range = X_RANGE
            y_range = Y_RANGE

            # Recreate the grid map from positions array
            grid_map_recreate = np.full((y_range, x_range), min_height)

            # Fill the grid map with values from positions array
            for pos in points:
                col, row, height = pos
                grid_map_recreate[int(row), int(col)] = height
    
    return grid_map_recreate

def load_points_grid_map_BB (csv_file):
    """Load bounding box vertices from a CSV file."""
    points = []
    heights = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            coordinates = [ [ float(row[i]), float(row[i+1]) ] for i in range(2, 10, 2)]
            height = (float(row[10], float(row[11])))
            points.append(coordinates)
            heights.append(height)
    np_points = np.array(points)
    np_heights = np.array(heights)
    return np_points, np_heights

def generate_grid_map_BB (grid_map_path):
    grid_map_files = [f for f in os.listdir(grid_map_path) if f.endswith('.cvs')]

    for file in grid_map_files:
            print(file)
            grid_map_path = os.path.join(grid_map_path, file)
            print(f"Loading {file}...")
            points, heights = load_points_grid_map_BB(grid_map_path)
            print(points)

            min_height = MIN_HEIGHT
            x_range = X_RANGE
            y_range = Y_RANGE

            # Recreate the grid map from positions array
            grid_map_recreate = np.full((y_range, x_range, 2), (min_height,min_height))

            # Fill the grid map with values from positions array
            for pos,i in enumerate(points):
                col, row = pos
                grid_map_recreate[int(row), int(col), 0] = heights[i][0]
                grid_map_recreate[int(row), int(col), 1] = heights[i][1]
    
    return grid_map_recreate


if __name__ == "__main__":

    grid_map_cross_s = generate_grid_map(LIDAR_CROSS_S_GRID_DIRECTORY)
    grid_map_BB_cross_s = generate_grid_map_BB(NEW_POSITIONS_LIDAR_CROSS_S_GRID_DIRECTORY)


    shape_input = (900,900,1)
    shape_output = (900,900,2)
    # Input layer for the 2.5D grid map
    grid_map_input = tfkl.Input(shape_input, name='grid_map_input')

    # Define the CNN architecture
    x = tfkl.Conv2D(32, (3, 3), activation='relu')(grid_map_input)
    x = tfkl.MaxPooling2D((2, 2))(x)
    x = tfkl.Conv2D(64, (3, 3), activation='relu')(x)
    x = tfkl.MaxPooling2D((2, 2))(x)
    x = tfkl.Conv2D(128, (3, 3), activation='relu')(x)
    x = tfkl.MaxPooling2D((2, 2))(x)
    x = tfkl.Flatten()(x)
    x = tfkl.Dense(256, activation='relu')(x)

    # Output layer to predict bounding box positions
    bounding_boxes_output = tfkl.Dense(shape_output, activation='linear', name='bounding_boxes_output')(x)

    # Create the model
    model = tfkl.Model(inputs=grid_map_input, outputs=bounding_boxes_output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Print the model summary
    model.summary()
