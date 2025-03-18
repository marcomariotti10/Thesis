import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import numpy as np
import open3d as o3d
import os
import csv
import sys

def create_point_cloud_from_grid_map(grid_map):
    """Create a point cloud from the grid map."""
    points = []
    for y in range(grid_map.shape[0]):
        for x in range(grid_map.shape[1]):
            z = grid_map[y, x]
            points.append([x, y, z])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    return point_cloud

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
            # Extract the 3D coordinates of the 8 bounding box vertices
            vertices = [
                [float(row[i]), float(row[i + 1]), float(row[i + 2])] for i in range(2, 12, 3)
            ] + [
                [float(row[i]), float(row[i + 1]), min_height] for i in range(2, 12, 3)
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

def show_grid_map(grid_map_directory, BB_directory, specific_csv):
    # Load point cloud from .ply file
    grid_map_files = sorted([f for f in os.listdir(grid_map_directory) if f.endswith('.csv')])
    #BB_files = sorted([f for f in os.listdir(BB_directory) if f.endswith('.csv')])

    if (specific_csv >= 0 and specific_csv < len(grid_map_files)):
        print(f"The index is: {specific_csv}")
        grid_map_files = grid_map_files[specific_csv:]
        #BB_files = BB_files[specific_csv:]
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

        #BB_file = BB_files[i]
        #BB_path = os.path.join(BB_directory, BB_file)
        #print(f"Loading {BB_file}...")
        #bounding_box_vertices = load_bounding_box(BB_path)

        # Create Open3D Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=1920, height=1200, left=0, top=80, visible=True)

        # Create a LineSet (bounding box)
        #for vertices in bounding_box_vertices:
            #box_lines = create_bounding_box_lines(vertices)
            #vis.add_geometry(box_lines)

        # Create point cloud from grid map
        point_cloud = create_point_cloud_from_grid_map(grid_map_recreate)

        # Create points from the grid map
        points = []
        for i in range(grid_map_recreate.shape[0]):
            for j in range(grid_map_recreate.shape[1]):
                z = grid_map_recreate[i, j]
                points.append([i, j, z])

        # Add point cloud to visualizer
        vis.add_geometry(point_cloud)

        # Update the visualizer to set the window dimensions
        vis.update_geometry(point_cloud)
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
            path_lidar = LIDAR_X_GRID_DIRECTORY.replace('X', user_input)
            new_position_path = POSITION_LIDAR_X_GRID_NO_BB.replace('X', user_input)
            lidar_file = LIDAR_FILE_X[int(user_input)-1]
            break
        else:
            print("Invalid input.")

    show_grid_map(path_lidar, new_position_path, lidar_file)