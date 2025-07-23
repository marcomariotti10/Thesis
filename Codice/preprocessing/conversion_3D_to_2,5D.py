import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import open3d as o3d
import numpy as np
import os
from scipy.stats import mode
import time
from multiprocessing import Pool

def process_file(i,file, ply_directory, folder_path):
    # Extract file
    #print(f"Processing file {file}...")
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

    filename = f"{file[:-4]}_{i}.csv"
    file_path = os.path.join(folder_path, filename)
    np.savetxt(file_path, positions, delimiter=",", fmt=['%d', '%d', '%.2f'])

def generate_grid_map(ply_directory, folder_path, lidar_number):

    # Replace 'X' in the paths with the lidar_number
    ply_directory = ply_directory.replace('X', str(lidar_number))
    folder_path = folder_path.replace('X', str(lidar_number))

    ply_files = sorted([f for f in os.listdir(ply_directory) if f.endswith('.ply')])

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with Pool() as pool:
        pool.starmap(process_file, [(i,file, ply_directory, folder_path) for i,file in enumerate(ply_files)])

if __name__ == "__main__":

    while True:
        user_input = input("Enter the number of the lidar for the single lidar, or enter 'all' to process all the lidar: ")
        if user_input == 'all':
            for i in range(NUMBER_OF_SENSORS):
                generate_grid_map(LIDAR_X_DIRECTORY, LIDAR_X_GRID_DIRECTORY, i+1)
                print("lidar" + str(i+1) + " done")
            break
        elif (int(user_input) in range(1, NUMBER_OF_SENSORS+1)):
            generate_grid_map(LIDAR_X_DIRECTORY, LIDAR_X_GRID_DIRECTORY, int(user_input))
            break
        else:
            print("Invalid input.")