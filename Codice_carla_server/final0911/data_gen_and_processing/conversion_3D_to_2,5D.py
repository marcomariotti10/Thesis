import open3d as o3d
import numpy as np
import os
from scipy.stats import mode
from constants import *

def generate_grid_map(ply_directory, folder_path):
    # Load point cloud from .ply file
    ply_files = sorted([f for f in os.listdir(ply_directory) if f.endswith('.ply')])

    grid_resolution = GRID_RESOLUTION  # Define the resolution of the grid
    x_min = X_MIN
    y_min = Y_MIN
    x_range = X_RANGE
    y_range = Y_RANGE

    for file in ply_files:
        # Extract file
        ply_path = os.path.join(ply_directory, file)
        #print(f"Loading {file}...")
        pcd = o3d.io.read_point_cloud(ply_path)
        # Extract points as numpy array
        points = np.asarray(pcd.points)

        # Define grid parameters
        # x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        # y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        # Calculate grid dimensions
        # x_range = int((x_max - x_min) / grid_resolution) + 1
        # y_range = int((y_max - y_min) / grid_resolution) + 1

        # Use this fixed value to have the same dimension for all the data

        # Initialize the grid map
        grid_map = np.full((y_range, x_range), FLOOR_HEIGHT, dtype=float)

        for point in points:
            x, y, z = point
            if (x > -x_min and x < x_min and y > -y_min and y < y_min):
                x_idx = int((x - x_min) / grid_resolution) # Because we don't add the +1 to the map dimensions (otherwise the NN can't take it) the values at 20m are put at 0m, but this is negligible.
                y_idx = int((y - y_min) / grid_resolution)
                grid_map[y_idx, x_idx] = max(grid_map[y_idx, x_idx], (z / grid_resolution))  # Take the maximum height divided for the grid_resolution to maintain the proportions

        #mode_height = mode(points[:, 2])[0][0]
        #grid_map[grid_map == -np.inf] = (mode_height /grid_resolution)

        # Find the indices where grid_map values are different from the minimum value
        non_zero_indices = np.nonzero(grid_map != FLOOR_HEIGHT)

        # Extract the values at these indices
        values = grid_map[non_zero_indices]

        # Combine the indices and values into a structured array
        positions = np.column_stack((non_zero_indices[1], non_zero_indices[0], values))
        
        # Define the desired folder and filename
        filename = file[:-4]
        filename = filename + ".csv"

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Full path to save the CSV file
        file_path = os.path.join(folder_path, filename)

        # Save the grid map to the specified folder
        np.savetxt(file_path, positions, delimiter=",", fmt=['%d', '%d', '%.2f'])

        #print(f"Grid map saved to {file_path}")


if __name__ == "__main__":

    path_lidar_1 = LIDAR_1_DIRECTORY
    new_path_1_grid = LIDAR_1_GRID_DIRECTORY

    path_lidar_2 = LIDAR_2_DIRECTORY
    new_path_2_grid = LIDAR_2_GRID_DIRECTORY

    path_lidar_3 = LIDAR_3_DIRECTORY
    new_path_3_grid = LIDAR_3_GRID_DIRECTORY

    while True:
        user_input = input("Enter 1 for lidar1, 2 for lidar2, 3 for lidar3, 4 for all: ")
        if user_input == '1':
            generate_grid_map(path_lidar_1, new_path_1_grid)
            break
        elif user_input == '2':
            generate_grid_map(path_lidar_2, new_path_2_grid)
            break
        elif user_input == '3':
            generate_grid_map(path_lidar_3, new_path_3_grid)
            break
        elif user_input == '4':
            generate_grid_map(path_lidar_1, new_path_1_grid)
            generate_grid_map(path_lidar_2, new_path_2_grid)
            generate_grid_map(path_lidar_3, new_path_3_grid)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3 or 4.")




