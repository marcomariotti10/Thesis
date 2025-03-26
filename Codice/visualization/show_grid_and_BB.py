import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import os
import matplotlib.pyplot as plt
import sys
import torch
import ast
import numpy as np
import csv

def visualize_data(grid_map, grid_map_BB):

    #print(f"Images shape: {grid_map.shape}")
    #print(f"Labels shape: {grid_map_BB.shape}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_map, cmap='gray', alpha=0.5)
    ax.imshow(grid_map_BB, cmap='jet', alpha=0.5)
    ax.set_title('Overlay of Original and Prediction Grid Maps')
    plt.show()

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

def load_bounding_box(csv_file):
    """Load bounding box vertices from a CSV file."""
    min_height = MIN_HEIGHT
    bounding_box_vertices = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            vertices = [row[2]]
            bounding_box_vertices.append(vertices)
    bounding_boxes = np.array(bounding_box_vertices)
    return bounding_boxes

def show_grid_map(grid_map_directory, BB_directory, specific_csv):
    # Load point cloud from .ply file
    grid_map_files = sorted([f for f in os.listdir(grid_map_directory) if f.endswith('.csv')])
    BB_files = sorted([f for f in os.listdir(BB_directory) if f.endswith('.csv')])

    if (specific_csv >= 0 and specific_csv < len(grid_map_files)):
        print(f"The index is: {specific_csv}")
        grid_map_files = grid_map_files[specific_csv:]
        BB_files = BB_files[specific_csv:]
    else:
        print(f"ERROR : {specific_csv} is not correct")

    for i,file in enumerate(grid_map_files):
        grid_map_path = os.path.join(grid_map_directory, file)
        print(f"Loading {file}...")
        points = load_points_grid_map(grid_map_path)

        min_height = MIN_HEIGHT
        x_range = X_RANGE
        y_range = Y_RANGE

        # Recreate the grid map from positions array
        grid_map_recreate = np.full((y_range, x_range), FLOOR_HEIGHT, dtype=float)

        # Fill the grid map with values from positions array
        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = height

        BB_file = BB_files[i]
        BB_path = os.path.join(BB_directory, BB_file)
        print(f"Loading {BB_file}...")
        bounding_box_vertices = load_bounding_box(BB_path)
        #print(f"shape: {bounding_box_vertices.shape}")
        # Initialize a list to store all pairs
        all_pairs = []

        # Iterate through each row in the numpy array
        for row in bounding_box_vertices:
            # Extract the string from the array
            string_data = row[0]
            # Safely evaluate the string to convert it into a list of tuples
            pairs = ast.literal_eval(string_data)
            # Add the pairs to the all_pairs list
            all_pairs.extend(pairs)

        grid_map_recreate_BB = np.full((y_range, x_range), 0, dtype=float)
        # Fill the grid map with values from positions array
        for pair in all_pairs:
            col, row= pair
            grid_map_recreate_BB[int(row), int(col)] = 1
        
        visualize_data(grid_map_recreate, grid_map_recreate_BB)
       

if __name__ == "__main__":

    while True:
        user_input = input("Enter the number of the lidar: ")
        if (int(user_input) in range(1, NUMBER_OF_SENSORS+1)):

            # Replace 'X' in the paths with the lidar_number
            path_lidar = LIDAR_X_GRID_DIRECTORY.replace('X', user_input)
            new_position_path = POSITION_LIDAR_X_GRID.replace('X', user_input)
            lidar_file = LIDAR_FILE_X[int(user_input)-1]
            break
        else:
            print("Invalid input.")

    show_grid_map(path_lidar, new_position_path, lidar_file)