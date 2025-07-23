import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import datetime
import sys
import os
import shutil
import open3d as o3d
import csv
import numpy as np
import math

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

def load_bounding_box(csv_file):
    """Load bounding box vertices from a CSV file."""
    bounding_box_vertices = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row[1] == 'pedestrian':
                increment = INCREMENT_BB_PEDESTIAN
            else:
                increment = 0.0
            # Extract the 3D coordinates of the 8 bounding box vertices
            center = [
                float(row[2]), float(row[3]), float(row[4])
            ]
            dimension = [
                float(row[5]) + increment, float(row[6]) + increment, float(row[7]) + increment
            ]
            rotation = [float(row[8]), float(row[9]), float(row[10])
                        ]
            offsets = [
                [dimension[0], dimension[1], dimension[2]],
                [dimension[0], -dimension[1], dimension[2]],
                [-dimension[0], dimension[1], dimension[2]],
                [-dimension[0], -dimension[1], dimension[2]],
                [dimension[0], dimension[1], -dimension[2]],
                [dimension[0], -dimension[1], -dimension[2]],
                [-dimension[0], dimension[1], -dimension[2]],
                [-dimension[0], -dimension[1], -dimension[2]]
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
            
            bounding_box_vertices.append(vertices)
    bounding_boxes = np.array(bounding_box_vertices)
    return bounding_boxes

def create_bounding_box_lines(vertices):
    """Create a LineSet representing the bounding box from the 8 vertices."""
    lines = [
        [0, 1], [1, 3], [3, 2], [2, 0],  # Four edges of the bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # Four edges of the top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Four vertical edges
    ]
        
    # Create Open3D LineSet for visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set

def visualize_ply_with_bounding_boxes(ply_directory, csv_directory, specific_ply):
    """Visualize point clouds with bounding boxes."""
    ply_files = sorted([f for f in os.listdir(ply_directory) if f.endswith('.ply')])
    csv_files = sorted([f for f in os.listdir(csv_directory) if f.endswith('.csv')])

    if len(ply_files) != len(csv_files):
        print(f"Warning: The number of .ply files ({len(ply_files)}) and .csv files ({len(csv_files)}) do not match!")
        return
    if (specific_ply >= 0 and specific_ply < len(ply_files)):
        print(f"The index is: {specific_ply}")
        ply_files = ply_files[specific_ply:]
        csv_files = csv_files[specific_ply:]
    else:
        print(f"ERROR : {specific_ply} is not correct")
    
    for i, ply_file in enumerate(ply_files) : 
        # Create Open3D Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=1920, height=1200, left=0, top=80, visible=True)

        # Load .ply file (point cloud)
        ply_path = os.path.join(ply_directory, ply_file)
        print(f"Loading {ply_file}...")
        cloud = o3d.io.read_point_cloud(ply_path)

        # Load bounding box vertices from .csv file
        csv_file = csv_files[i] 
        csv_path = os.path.join(csv_directory, csv_file)
        print(f"Loading {csv_file}...")
        bounding_box_vertices = load_bounding_box(csv_path)

        # Create a LineSet (bounding box)
        for vertices in bounding_box_vertices:
            box_lines = create_bounding_box_lines(vertices)
            vis.add_geometry(box_lines)

        # Add point cloud to visualizer
        vis.add_geometry(cloud)

        # Update the visualizer to set the window dimensions
        vis.update_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()

        # Run the visualizer
        vis.run()

        # Close the visualizer after the user closes the window
        vis.destroy_window()

if __name__ == "__main__":

    while True:
        user_input = input("Enter the number of the lidar: ")
        if (int(user_input) in range(1, NUMBER_OF_SENSORS+1)):

            # Replace 'X' in the paths with the lidar_number
            path_lidar = LIDAR_X_DIRECTORY.replace('X', user_input)
            new_position_path = SNAPSHOT_X_DIRECTORY.replace('X', user_input)
            lidar_file = LIDAR_FILE_X[int(user_input)-1]
            break
        else:
            print("Invalid input.")

    visualize_ply_with_bounding_boxes(path_lidar, new_position_path, lidar_file)


