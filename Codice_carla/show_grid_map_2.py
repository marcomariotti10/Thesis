import numpy as np
import open3d as o3d
import os
import csv
import sys
from constants import *


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
    print(np_points)
    return np_points

def load_bounding_box(csv_file):
    """Load bounding box vertices from a CSV file."""
    bounding_box_vertices = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            vertices = [
                [float(row[i]), float(row[i + 1]), float(row[10])] for i in range(2, 10, 2)
            ] + [
                [float(row[i]), float(row[i + 1]), float(row[11])] for i in range(2, 10, 2)
            ]
            print(vertices)
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

def show_grid_map(grid_map_directory, BB_directory, specific_csv):
 # Load point cloud from .ply file
    grid_map_files = [f for f in os.listdir(grid_map_directory) if f.endswith('.csv')]
    BB_files = [f for f in os.listdir(BB_directory) if f.endswith('.csv')]

    if specific_csv in grid_map_files:
        index = grid_map_files.index(specific_csv)
        print(f"The index of {specific_csv} is: {index}")
        grid_map_files = grid_map_files[index:]
        BB_files = BB_files[index:]
    else:
        print(f"ERROR : {specific_csv} is not in the list")

    for i, file in enumerate(grid_map_files):
        grid_map_path = os.path.join(grid_map_directory, file)
        print(f"Loading {file}...")
        points = load_points_grid_map(grid_map_path)
        print(points)

        min_height = MIN_HEIGHT
        x_range = X_RANGE
        y_range = Y_RANGE

        # Recreate the grid map from positions array
        grid_map_recreate = np.full((y_range, x_range), min_height)

        # Fill the grid map with values from positions array
        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = height

        BB_files = BB_files[i]
        BB_path = os.path.join(BB_directory, BB_files)
        print(f"Loading {BB_files}...")
        bounding_box_vertices = load_bounding_box(BB_path)

        # Create a LineSet (bounding box)
        line_sets = []
        for vertices in bounding_box_vertices:
            box_lines = create_bounding_box_lines(vertices)
            line_sets.append(box_lines)

        # Create point cloud from grid map
        point_cloud = create_point_cloud_from_grid_map(grid_map_recreate)

        # Visualize point cloud and bounding boxes
        o3d.visualization.draw_geometries([point_cloud] + line_sets)


if __name__ == "__main__":

    path_grid_map_lidar_cross_s = LIDAR_CROSS_S_GRID_DIRECTORY
    path_grid_BB_lidar_cross_s = NEW_POSITIONS_LIDAR_CROSS_S_GRID_DIRECTORY

    show_grid_map(path_grid_map_lidar_cross_s, path_grid_BB_lidar_cross_s, GRID_FILE_CROSS_S)
