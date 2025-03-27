import sys
import os
import threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import numpy as np
import pandas as pd
import csv
import math

def rotate_point(point, rotation_matrix):
    x, y, z = point
    return (
        rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z,
        rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z,
        rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z
    )

def get_rotation_matrix(pitch, roll, yaw):
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    yaw = math.radians(yaw)
    
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
    
    rot_final = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rot_final[i][j] = sum(rot_x[i][k] * rot_y[k][j] for k in range(3))
    
    rot_final_final = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rot_final_final[i][j] = sum(rot_final[i][k] * rot_z[k][j] for k in range(3))
    
    return rot_final_final

def process_csv_file(BB_path, file, folder_path):
    grid_resolution = GRID_RESOLUTION
    x_min = X_MIN + (int(INCREASE_GRID_RANGE/2)*grid_resolution)
    y_min = Y_MIN + (int(INCREASE_GRID_RANGE/2)*grid_resolution)
    x_range = X_RANGE
    y_range = Y_RANGE

    reader = csv.reader(open(os.path.join(BB_path, file)))

    bounding_box_vertices = []
    bounding_box_ids = []
    bounding_box_labels = []
    all_positions = []

    next(reader)  # Skip header

    for row in reader:
        if row[1] == 'pedestrian':
            increment = INCREMENT_BB_PEDESTIAN
        else:
            increment = 0.0
        center = [float(row[2]), float(row[3]), float(row[4])]
        dimension = [float(row[5]) + increment, float(row[6]) + increment, float(row[7]) + increment]
        rotation = [float(row[8]), float(row[9]), float(row[10])]
        offsets = [
            [dimension[0], dimension[1], dimension[2]],
            [dimension[0], -dimension[1], dimension[2]],
            [-dimension[0], dimension[1], dimension[2]],
            [-dimension[0], -dimension[1], dimension[2]],
        ]
        rotation_matrix = get_rotation_matrix(rotation[0], rotation[1], rotation[2])
        vertices = [
            (
                center[0] + rotate_point(offset, rotation_matrix)[0],
                center[1] + rotate_point(offset, rotation_matrix)[1],
                center[2] + rotate_point(offset, rotation_matrix)[2]
            )
            for offset in offsets
        ]

        x_old = 10000
        y_old = 10000
        z_old = 10000
        outside_count = 0
        all_in_range = True
        all_same = False
        for ver in vertices:
            if ((ver[0] > X_MIN - REDUCING_RANGE) or (ver[0] < -X_MIN + REDUCING_RANGE) or (ver[1] > Y_MIN - REDUCING_RANGE) or (ver[1] < -Y_MIN + REDUCING_RANGE)):
                outside_count += 1
                if outside_count >= 4:
                    all_in_range = False
                    break
            if ((ver[0] != x_old) or (ver[1] != y_old) or (ver[2] != z_old)):
                x_old = ver[0]
                y_old = ver[1]
                z_old = ver[2]
            else:
                all_same = True
                break

        if all_in_range and not all_same:
            bounding_box_vertices.append(vertices)
            bounding_box_ids.append(row[0])
            bounding_box_labels.append(row[1])

    for vertic in bounding_box_vertices:
        grid_map = np.full((y_range + INCREASE_GRID_RANGE, x_range + INCREASE_GRID_RANGE), 0, dtype=float)
        grid_vertices = []
        for point in vertic:
            x, y, z = point
            x_idx = int((x - x_min) / grid_resolution)
            y_idx = int((y - y_min) / grid_resolution)
            grid_map[y_idx, x_idx] = z
        
        grid_vertices = np.nonzero(grid_map != 0)
        positions_array = np.column_stack((grid_vertices[1], grid_vertices[0]))
        positions = [tuple(row) for row in positions_array.tolist()]
        position = np.array(positions)
        #print(f"position: {position}")
        height_BB = 1  # Assuming all vertices have the same height
        fill_polygon(grid_map, position, height_BB)

        increase_grid_range_half = int(INCREASE_GRID_RANGE / 2)
        smaller_grid = grid_map[(increase_grid_range_half):(increase_grid_range_half + y_range), (increase_grid_range_half):(increase_grid_range_half + x_range)]

        non_zero_indices = np.nonzero(smaller_grid != 0)
        positions_array = np.column_stack((non_zero_indices[1], non_zero_indices[0]))
        positions = [tuple(row) for row in positions_array.tolist()]
        all_positions.append(positions)

    data = [
        {
            "actor_id": actor_id,
            "label": label,
            "points": vert
        }
        for actor_id, label, vert in zip(bounding_box_ids, bounding_box_labels, all_positions)
    ]

    path = os.path.join(folder_path, file)
    os.makedirs(folder_path, exist_ok=True)

    with open(path, 'w', newline='') as file:
        fieldnames = [
            'actor_id', 'label',
            'points'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def convert_BB_into_25D(BB_path, folder_path, lidar_number):
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
    while True:
        user_input = input("Enter the number of the lidar for the single lidar, or enter 'all' to process all the lidar: ")
        if user_input == 'all':
            threads = []
            for i in range(NUMBER_OF_SENSORS):
                thread = threading.Thread(target=convert_BB_into_25D, args=(NEW_POSITION_LIDAR_X_DIRECTORY, POSITION_LIDAR_X_GRID, i+1))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            print("Processing completed for all sensors.")
            break
    
        elif user_input.isdigit() and (1 <= int(user_input) <= NUMBER_OF_SENSORS):
            convert_BB_into_25D(NEW_POSITION_LIDAR_X_DIRECTORY, POSITION_LIDAR_X_GRID, int(user_input))
            break
        
        else:
            print("Invalid input.")