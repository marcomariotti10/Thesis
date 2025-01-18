import open3d as o3d
import numpy as np
import os
from constants import *

def generate_grid_map(ply_directory, folder_path):
    # Load point cloud from .ply file
    ply_files = [f for f in os.listdir(ply_directory) if f.endswith('.ply')]

    for file in ply_files:
        # Extract file
        ply_path = os.path.join(ply_directory, file)
        print(f"Loading {file}...")
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
        grid_resolution = GRID_RESOLUTION  # Define the resolution of the grid
        x_min = 45
        y_min = 45
        x_range = X_RANGE
        y_range = Y_RANGE

        # Initialize the grid map
        grid_map = np.full((y_range, x_range), -np.inf)

        for point in points:
            x, y, z = point
            x_idx = int((x - x_min) / grid_resolution)
            y_idx = int((y - y_min) / grid_resolution)
            grid_map[y_idx, x_idx] = max(grid_map[y_idx, x_idx], (z / grid_resolution))  # Take the maximum height divided for the grid_resolution to maintain the proportions

        min_height = np.min(points[:, 2])
        grid_map[grid_map == -np.inf] = (min_height /grid_resolution)

        # Find the indices where grid_map values are different from the minimum value
        non_zero_indices = np.nonzero(grid_map != (min_height/grid_resolution))

        # Extract the values at these indices
        values = grid_map[non_zero_indices]

        # Combine the indices and values into a structured array
        positions = np.column_stack((non_zero_indices[1], non_zero_indices[0], values))

        print(positions)

        
        # Define the desired folder and filename
        filename = file[:-4]
        filename = filename + ".csv"

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Full path to save the CSV file
        file_path = os.path.join(folder_path, filename)

        # Save the grid map to the specified folder
        np.savetxt(file_path, positions, delimiter=",")

        print(f"Grid map saved to {file_path}")


if __name__ == "__main__":

    path_lidar_output_cross_s = LIDAR_CROSS_S_DIRECTORY
    new_path_output_cross_s_grid = LIDAR_CROSS_S_GRID_DIRECTORY

    path_lidar_near_station_s = LIDAR_NEAR_STATION_S_DIRECTORY
    new_path_output_near_station_s_grid = LIDAR_NEAR_STATION_S_GRID_DIRECTORY

    path_lidar_int_road_s = LIDAR_INT_ROAD_S_DIRECTORY
    new_path_output_int_road_s_grid = LIDAR_INT_ROAD_S_GRID_DIRECTORY


    generate_grid_map(path_lidar_output_cross_s, new_path_output_cross_s_grid)
    generate_grid_map(path_lidar_near_station_s, new_path_output_near_station_s_grid)
    generate_grid_map(path_lidar_int_road_s, new_path_output_int_road_s_grid)



