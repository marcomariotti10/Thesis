import open3d as o3d
import numpy as np
import os
from scipy.stats import mode
from constants import *
import time
from multiprocessing import Pool

def process_file(file, ply_directory, folder_path):
    # Extract file
    ply_path = os.path.join(ply_directory, file)
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    grid_resolution = GRID_RESOLUTION
    x_min = X_MIN
    y_min = Y_MIN
    x_range = X_RANGE
    y_range = Y_RANGE

    grid_map = np.full((y_range, x_range), FLOOR_HEIGHT, dtype=float)

    valid_points = points[
        (points[:, 0] > -x_min) & (points[:, 0] < x_min) &
        (points[:, 1] > -y_min) & (points[:, 1] < y_min)
    ]

    x_indices = ((valid_points[:, 0] - x_min) / grid_resolution).astype(int)
    y_indices = ((valid_points[:, 1] - y_min) / grid_resolution).astype(int)
    z_values = valid_points[:, 2] / grid_resolution

    np.maximum.at(grid_map, (y_indices, x_indices), z_values)

    # Set the first row and first column to FLOOR_HEIGHT (because the grid is 400x400 and not 401x401)
    grid_map[0, :] = FLOOR_HEIGHT
    grid_map[:, 0] = FLOOR_HEIGHT

    non_zero_indices = np.nonzero(grid_map != FLOOR_HEIGHT)
    values = grid_map[non_zero_indices]
    positions = np.column_stack((non_zero_indices[1], non_zero_indices[0], values))

    filename = file[:-4] + ".csv"
    file_path = os.path.join(folder_path, filename)
    np.savetxt(file_path, positions, delimiter=",", fmt=['%d', '%d', '%.2f'])

def generate_grid_map(ply_directory, folder_path):
    ply_files = sorted([f for f in os.listdir(ply_directory) if f.endswith('.ply')])

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with Pool() as pool:
        pool.starmap(process_file, [(file, ply_directory, folder_path) for file in ply_files])

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
            print("Lidar 1 done")
            generate_grid_map(path_lidar_2, new_path_2_grid)
            print("Lidar 2 done")
            generate_grid_map(path_lidar_3, new_path_3_grid)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3 or 4.")




