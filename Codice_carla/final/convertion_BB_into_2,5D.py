import os
import numpy as np
import pandas as pd
import sys
import csv
import math
from constants import *


def rotate_point(point, rotation_matrix):
    x, y, z = point
    return (
        rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z,
        rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z,
        rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z
    )

def get_rotation_matrix(pitch, roll, yaw):
    # Convert degrees to radians
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    yaw = math.radians(yaw)
    
    # Rotation matrix for pitch (X-axis)
    rot_x = [
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ]
    
    # Rotation matrix for roll (Y-axis)
    rot_y = [
        [math.cos(roll), 0, math.sin(roll)],
        [0, 1, 0],
        [-math.sin(roll), 0, math.cos(roll)]
    ]
    
    # Rotation matrix for yaw (Z-axis)
    rot_z = [
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ]
    
    # Combined rotation matrix: Rot_final = Rot_x * Rot_y * Rot_z
    rot_final = [[0, 0, 0] for _ in range(3)]
    
    # Multiply Rot_x and Rot_y
    for i in range(3):
        for j in range(3):
            rot_final[i][j] = sum(rot_x[i][k] * rot_y[k][j] for k in range(3))
    
    # Multiply the result with Rot_z
    rot_final_final = [[0, 0, 0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            rot_final_final[i][j] = sum(rot_final[i][k] * rot_z[k][j] for k in range(3))
    
    return rot_final_final

def convert_BB_into_25D(BB_path, folder_path) :

    grid_resolution = GRID_RESOLUTION  # Define the resolution of the grid
    x_min = X_MIN
    y_min = Y_MIN
    x_range = X_RANGE
    y_range = Y_RANGE
    
    # Load point cloud from .csv file
    csv_files = [f for f in os.listdir(BB_path) if f.endswith('.csv')]

    for file in csv_files:
        print(file)
        reader = csv.reader(open(os.path.join(BB_path, file)))

        bounding_box_vertices = []
        bounding_box_ids = []
        bounding_box_labels = []
        all_positions = []

        next(reader)  # Skip header

        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            center = [
                float(row[2]), float(row[3]), float(row[4])
            ]
            dimension = [
                float(row[5]), float(row[6]), float(row[7])
            ]
            rotation = [float(row[8]), float(row[9]), float(row[10])
                        ]
            offsets = [
                [dimension[0], dimension[1], dimension[2]],
                [dimension[0], -dimension[1], dimension[2]],
                [-dimension[0], dimension[1], dimension[2]],
                [-dimension[0], -dimension[1], dimension[2]],
            ]
            # Get the rotation matrix for pitch, yaw, and roll
            rotation_matrix = get_rotation_matrix(rotation[0], rotation[1], rotation[2])
    
            # Apply the rotation to each offset to get the rotated vertices
            vertices = [
                (
                    center[0] + rotate_point(offset, rotation_matrix)[0],
                    center[1] + rotate_point(offset, rotation_matrix)[1],
                    center[2] + rotate_point(offset, rotation_matrix)[2]
                )
                for offset in offsets
            ]

            x_old = 10000 # The value is set to not generate error
            y_old = 10000
            z_old = 10000
            all_in_range = True
            all_same = False
            for ver in vertices:
                if ( (ver[0] > X_MIN) or (ver[0] < -X_MIN) or (ver[1] > X_MIN) or (ver[1] < -X_MIN) ): #Check if all the verteces are in the range of lidar
                    all_in_range = False
                    break
                if ( (ver[0] != x_old) or (ver[1] != y_old) or (ver[2] != z_old) ): #Check if the vertices are not equal, because some bounding box can have all the vertex equal (probably the actor hasn't spawn)
                    x_old = ver[0]
                    y_old = ver[1]
                    z_old = ver[2]
                else: 
                    all_same = True
                    break

            if all_in_range and not all_same: #Save the vertices and the rest of information anly if all the requirement are satisfied
                bounding_box_vertices.append(vertices)
                bounding_box_ids.append(row[0])
                bounding_box_labels.append(row[1])

                # Initialize the grid map
                grid_map = np.full((y_range, x_range), -np.inf)

                for vertic in bounding_box_vertices:
                    for point in vertic:
                        x, y, z = point
                        x_idx = int((x - x_min) / grid_resolution)
                        y_idx = int((y - y_min) / grid_resolution)
                        grid_map[y_idx, x_idx] = max(grid_map[y_idx, x_idx], int(z / grid_resolution))  # Take the maximum height divided for the grid_resolution to maintain the proportions

                non_zero_indices = np.nonzero(grid_map != -np.inf)
                
                # Extract the values at these indices
                values = grid_map[non_zero_indices]

                # Combine the indices and values into a structured array
                positions_array = np.column_stack((non_zero_indices[0], non_zero_indices[1], values))
                
                # Convert the structured array to a list of tuples
                positions = [tuple(row) for row in positions_array.tolist()]

                all_positions.append(positions)

        data = [{"actor_id": actor_id, "label" : label,"vertice_1x" : vert[0][1] ,"vertice_1y" : vert[0][0], "vertice_2x" : vert[1][1],"vertice_2y" : vert[1][0], "vertice_3x" : vert[2][1], "vertice_3y" : vert[2][0], "vertice_4x" : vert[3][1],"vertice_4y" : vert[3][0], "height" : vert[0][2]} for actor_id, label, vert in zip(bounding_box_ids , bounding_box_labels, all_positions) ]
        
        path = os.path.join(folder_path, file)

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        with open(path, 'w', newline='') as file:
            fieldnames = ['actor_id', 'label', 'vertice_1x','vertice_1y', 'vertice_2x', 'vertice_2y', 'vertice_3x', 'vertice_3y', 'vertice_4x', 'vertice_4y', 'height']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # Write the header 
            writer.writeheader() 
            # Write the rows 
            writer.writerows(data)

if __name__ == "__main__":

    path_lidar_cross_s_positions = NEW_POSITION_LIDAR_CROSS_S_DIRECTORY
    path_BB_cros_s_grid = NEW_POSITIONS_LIDAR_CROSS_S_GRID_DIRECTORY

    path_lidar_near_station_s_positions = NEW_POSITION_LIDAR_NEAR_STATION_S_DIRECTORY
    path_BB_near_station_s_grid = NEW_POSITIONS_LIDAR_NEAR_STATION_S_GRID_DIRECTORY

    path_lidar_int_road_s_positions = NEW_POSITION_LIDAR_INT_ROAD_S_DIRECTORY
    path_BB_int_road_s_grid = NEW_POSITIONS_LIDAR_INT_ROAD_S_GRID_DIRECTORY

    convert_BB_into_25D(path_lidar_cross_s_positions, path_BB_cros_s_grid)
    #convert_BB_into_25D(path_lidar_near_station_s_positions, path_BB_near_station_s_grid)
    #convert_BB_into_25D(path_lidar_int_road_s_positions, path_BB_int_road_s_grid)
