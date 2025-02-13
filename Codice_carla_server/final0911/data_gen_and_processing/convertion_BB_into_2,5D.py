import os
import numpy as np
import pandas as pd
import sys
import csv
import math
from constants import *
from multiprocessing import Pool

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

def process_csv_file(args):
    BB_path, file, folder_path = args
    grid_resolution = GRID_RESOLUTION
    x_min = X_MIN
    y_min = Y_MIN
    x_range = X_RANGE
    y_range = Y_RANGE

    reader = csv.reader(open(os.path.join(BB_path, file)))

    bounding_box_vertices = []
    bounding_box_ids = []
    bounding_box_labels = []
    all_positions = []

    next(reader)  # Skip header

    for row in reader:
        center = [float(row[2]), float(row[3]), float(row[4])]
        dimension = [float(row[5]), float(row[6]), float(row[7])]
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
                if outside_count >= 3:
                    all_in_range = False
                    break
            if ((ver[0] != x_old) or (ver[1] != y_old) or (ver[2] != z_old)):
                x_old = ver[0]
                y_old = ver[1]
                z_old = ver[2]
            else:
                all_same = True
                break
        
        # This condition to eliminate all the bounding box partially outside the grid that are also rotated (too complex to manage)
        if outside_count == 1 or outside_count == 2:
            if (vertices[0][0] >= vertices[1][0] + RANGE_FOR_ROTATED_VEHICLES) or (vertices[0][0] <= vertices[1][0] - RANGE_FOR_ROTATED_VEHICLES):
                all_in_range = False

        if all_in_range and not all_same:
            bounding_box_vertices.append(vertices)
            bounding_box_ids.append(row[0])
            bounding_box_labels.append(row[1])

    for vertic in bounding_box_vertices:
        grid_map = np.full((y_range, x_range), FLOOR_HEIGHT, dtype=float)
        for point in vertic:
            x, y, z = point
            x_idx = int((x - x_min) / grid_resolution)
            if x_idx >= 0:
                x_idx = -1
            if x_idx < -X_RANGE:
                x_idx = 0
            y_idx = int((y - y_min) / grid_resolution)
            if y_idx >= 0:
                y_idx = -1
            if y_idx < -Y_RANGE:
                y_idx = 0
            grid_map[y_idx, x_idx] = max(grid_map[y_idx, x_idx], (z / grid_resolution))

        non_zero_indices = np.nonzero(grid_map != FLOOR_HEIGHT)
        values = grid_map[non_zero_indices]
        positions_array = np.column_stack((non_zero_indices[0], non_zero_indices[1], values))
        positions = [tuple(row) for row in positions_array.tolist()]
        all_positions.append(positions)

    data = [
        {
            "actor_id": actor_id,
            "label": label,
            "x1": vert[0][1], "y1": vert[0][0], "h1": f"{vert[0][2]:.2f}",
            "x2": vert[1][1], "y2": vert[1][0], "h2": f"{vert[1][2]:.2f}",
            "x3": vert[2][1], "y3": vert[2][0], "h3": f"{vert[2][2]:.2f}",
            "x4": vert[3][1], "y4": vert[3][0], "h4": f"{vert[3][2]:.2f}"
        }
        for actor_id, label, vert in zip(bounding_box_ids, bounding_box_labels, all_positions)
    ]

    path = os.path.join(folder_path, file)
    os.makedirs(folder_path, exist_ok=True)

    with open(path, 'w', newline='') as file:
        fieldnames = [
            'actor_id', 'label',
            'x1', 'y1', 'h1',
            'x2', 'y2', 'h2',
            'x3', 'y3', 'h3',
            'x4', 'y4', 'h4'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def convert_BB_into_25D(BB_path, folder_path):
    csv_files = sorted([f for f in os.listdir(BB_path) if f.endswith('.csv')])
    with Pool() as pool:
        pool.map(process_csv_file, [(BB_path, file, folder_path) for file in csv_files])

if __name__ == "__main__":
    path_lidar_1_positions = NEW_POSITION_LIDAR_1_DIRECTORY
    path_BB_1_grid = NEW_POSITIONS_LIDAR_1_GRID_DIRECTORY

    path_lidar_3_positions = NEW_POSITION_LIDAR_3_DIRECTORY
    path_BB_3_grid = NEW_POSITIONS_LIDAR_3_GRID_DIRECTORY

    path_lidar_2_positions = NEW_POSITION_LIDAR_2_DIRECTORY
    path_BB_2_grid = NEW_POSITIONS_LIDAR_2_GRID_DIRECTORY

    while True:
        user_input = input("Enter 1 for lidar1, 2 for lidar2, 3 for lidar3, 4 for all: ")
        if user_input == '1':
            convert_BB_into_25D(path_lidar_1_positions, path_BB_1_grid)
            break
        elif user_input == '2':
            convert_BB_into_25D(path_lidar_2_positions, path_BB_2_grid)
            break
        elif user_input == '3':
            convert_BB_into_25D(path_lidar_3_positions, path_BB_3_grid)
            break
        elif user_input == '4':
            convert_BB_into_25D(path_lidar_1_positions, path_BB_1_grid)
            print("Lidar 1 done")
            convert_BB_into_25D(path_lidar_2_positions, path_BB_2_grid)
            print("Lidar 2 done")
            convert_BB_into_25D(path_lidar_3_positions, path_BB_3_grid)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3 or 4.")




