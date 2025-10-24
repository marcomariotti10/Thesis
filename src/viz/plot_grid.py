"""
LiDAR Grid Map 3D Visualization Tool

This module provides interactive 3D visualization capabilities for LiDAR-generated grid maps
and associated bounding box data. It converts 2D grid maps into 3D point clouds and displays
them using Open3D for detailed spatial analysis and verification.

Key Features:
- 3D point cloud visualization from 2D grid maps
- Interactive Open3D viewer with zoom, pan, and rotation
- Support for multiple LiDAR sensor configurations
- Configurable visualization parameters and file selection
- Height-based spatial representation for occupancy analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import numpy as np
import open3d as o3d
import csv

def create_point_cloud_from_grid_map(grid_map):
    """
    Convert a 2D grid map into a 3D point cloud for visualization.
    
    This function transforms a 2D height grid into a 3D point cloud where each
    grid cell becomes a 3D point with (x, y, height) coordinates. This enables
    3D visualization of LiDAR occupancy data using Open3D.
    
    Args:
        grid_map (np.ndarray): 2D array representing height/occupancy values
                              Shape: (height, width) where values are height measurements
    
    Returns:
        o3d.geometry.PointCloud: Open3D point cloud object ready for visualization
                                Contains points with coordinates (x, y, z) where:
                                - x corresponds to grid column index
                                - y corresponds to grid row index  
                                - z corresponds to height value from grid_map
    """
    points = []
    
    # Convert each grid cell to a 3D point
    for y in range(grid_map.shape[0]):        # Iterate through rows
        for x in range(grid_map.shape[1]):    # Iterate through columns
            z = grid_map[y, x]                # Extract height value
            points.append([x, y, z])          # Create 3D point [x, y, height]
    
    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    return point_cloud

def load_points_grid_map(csv_file):
    """
    Load 3D point data from a CSV file for grid map reconstruction.
    
    This function reads processed LiDAR point data stored in CSV format,
    extracting the (x, y, z) coordinates needed for grid map reconstruction.
    Each row contains spatial coordinates and height information.
    
    Args:
        csv_file (str): Path to CSV file containing point data
                       Expected format: each row has at least 3 columns (x, y, z)
    
    Returns:
        np.ndarray: Array of 3D points with shape (num_points, 3)
                   Each row contains [x_coordinate, y_coordinate, height]
    """
    points = []
    
    # Read CSV file and extract coordinate information
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Extract 3D coordinates from CSV row
            coordinates = [
                float(row[0]),  # X coordinate
                float(row[1]),  # Y coordinate  
                float(row[2])   # Z coordinate (height)
            ]
            points.append(coordinates)
    
    # Convert to numpy array for efficient processing
    np_points = np.array(points)
    return np_points


def show_grid_map(grid_map_directory, specific_csv):
    """
    Display interactive 3D visualization of LiDAR grid maps with optional filtering.
    
    This function loads LiDAR grid map data from CSV files and creates an interactive
    3D point cloud visualization using Open3D. It processes grid-based LiDAR data
    by reconstructing the 3D scene from height-based grid maps, allowing users to
    visualize and inspect spatial data quality and coverage.
    
    Args:
        grid_map_directory (str): Path to directory containing grid map CSV files
                                 Files should be numbered/ordered for sequential processing
        BB_directory (str): Path to directory containing bounding box files (currently unused)
                           Reserved for future bounding box visualization integration
        specific_csv (int): Index for starting file selection
                           If >= 0 and < number_of_files: starts from specified index
                           If invalid: displays error and processes all files
    
    Returns:
        None: Function displays interactive visualization windows for each grid map
    
    """
    # Load point cloud from .ply file
    grid_map_files = sorted([f for f in os.listdir(grid_map_directory) if f.endswith('.csv')])

    if (specific_csv >= 0 and specific_csv < len(grid_map_files)):
        grid_map_files = grid_map_files[specific_csv:]
    else:
        print(f"ERROR : {specific_csv} is not correct")

    for i,file in enumerate(grid_map_files):
        grid_map_path = os.path.join(grid_map_directory, file)
        points = load_points_grid_map(grid_map_path)

        x_range = DIMENSION_X
        y_range = DIMENSION_Y

        # Recreate the grid map from positions array
        grid_map_recreate = np.full((y_range, x_range), NO_DATA_VALUE, dtype=float)

        # Fill the grid map with values from positions array
        for pos in points:
            col, row, height = pos
            grid_map_recreate[int(row), int(col)] = height

        # Create Open3D Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=1920, height=1200, left=0, top=80, visible=True)

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
    import sys
    
    # Check if arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python plot_grid.py <lidar_number> <starting_file_number>")
        print("Example: python plot_grid.py 1 50")
        sys.exit(1)
    
    try:
        lidar_number = int(sys.argv[1])
        starting_file = int(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be integers")
        sys.exit(1)
    
    # Replace 'X' in the paths with the lidar_number
    path_lidar = HEIGHTMAP_DIRECTORY.replace('X', str(lidar_number))
    
    # Launch visualization with selected sensor parameters
    show_grid_map(path_lidar, starting_file)