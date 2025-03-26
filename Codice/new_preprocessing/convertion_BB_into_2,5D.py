import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import os
import numpy as np
import pandas as pd
import sys
import csv
import threading
import math

def rotate_point(point, rotation_matrix):
    x, y, z = point
    return (
        rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z,
        rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z,
        rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z
    )

def get_rotation_matrix(pitch, roll, yaw):
    pitch, roll, yaw = map(math.radians, (pitch, roll, yaw))
    
    rot_x = [
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ]
    
    rot_y = [
        [math.cos(roll), 0, math.sin(roll)],
        [0, 1, 0],
        [-math.sin(roll), 0, math.cos(roll)]
    ]
    
    rot_z = [
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ]
    
    rot_final = np.matmul(rot_x, rot_y)
    return np.matmul(rot_final, rot_z)

def process_csv_file(BB_path, file, folder_path):
    """Process a single CSV file and save converted 2.5D bounding box data."""
    grid_resolution = GRID_RESOLUTION
    x_min = X_MIN + (int(INCREASE_GRID_RANGE / 2) * grid_resolution)
    y_min = Y_MIN + (int(INCREASE_GRID_RANGE / 2) * grid_resolution)
    x_range, y_range = X_RANGE, Y_RANGE

    reader = csv.reader(open(os.path.join(BB_path, file)))
    next(reader)  # Skip header

    bounding_box_data = []

    for row in reader:
        increment = INCREMENT_BB_PEDESTIAN if row[1] == 'pedestrian' else 0.0
        center = list(map(float, row[2:5]))
        dimension = list(map(lambda d: float(d) + increment, row[5:8]))
        rotation = list(map(float, row[8:11]))
        rotation_matrix = get_rotation_matrix(*rotation)

        offsets = [[dimension[0], dimension[1], dimension[2]],
                   [dimension[0], -dimension[1], dimension[2]],
                   [-dimension[0], dimension[1], dimension[2]],
                   [-dimension[0], -dimension[1], dimension[2]]]

        vertices = [(
            center[0] + rotate_point(offset, rotation_matrix)[0],
            center[1] + rotate_point(offset, rotation_matrix)[1],
            center[2] + rotate_point(offset, rotation_matrix)[2]
        ) for offset in offsets]

        all_positions = []
        grid_map = np.full((y_range + INCREASE_GRID_RANGE, x_range + INCREASE_GRID_RANGE), 0, dtype=float)

        for x, y, z in vertices:
            x_idx = int((x - x_min) / grid_resolution)
            y_idx = int((y - y_min) / grid_resolution)
            grid_map[y_idx, x_idx] = z

        non_zero_indices = np.nonzero(grid_map != 0)
        positions_array = np.column_stack((non_zero_indices[1], non_zero_indices[0]))
        positions = [tuple(row) for row in positions_array.tolist()]
        all_positions.append(positions)

        bounding_box_data.append({
            "actor_id": row[0],
            "label": row[1],
            "points": all_positions
        })

    os.makedirs(folder_path, exist_ok=True)
    with open(os.path.join(folder_path, file), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['actor_id', 'label', 'points'])
        writer.writeheader()
        writer.writerows(bounding_box_data)

def convert_BB_into_25D(BB_path, folder_path, lidar_number):
    """Process all CSV files for a given LiDAR sensor using threading."""
    BB_path = BB_path.replace('X', str(lidar_number))
    folder_path = folder_path.replace('X', str(lidar_number))

    csv_files = sorted([f for f in os.listdir(BB_path) if f.endswith('.csv')])

    threads = []
    for file in csv_files:
        thread = threading.Thread(target=process_csv_file, args=(BB_path, file, folder_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    user_input = input("Enter the LiDAR sensor number or 'all' to process all: ")

    if user_input.lower() == 'all':
        sensor_threads = []
        for i in range(1, NUMBER_OF_SENSORS + 1):
            thread = threading.Thread(target=convert_BB_into_25D, args=(NEW_POSITION_LIDAR_X_DIRECTORY, POSITION_LIDAR_X_GRID, i))
            sensor_threads.append(thread)
            thread.start()

        for thread in sensor_threads:
            thread.join()

        print("Processing completed for all sensors.")
    
    elif user_input.isdigit() and 1 <= int(user_input) <= NUMBER_OF_SENSORS:
        convert_BB_into_25D(NEW_POSITION_LIDAR_X_DIRECTORY, POSITION_LIDAR_X_GRID, int(user_input))

    else:
        print("Invalid input.")