"""
Point Cloud to 2.5D Grid Conversion Module

This module converts 3D point cloud data (PLY format) into 2.5D grid map representations.
The conversion process transforms sparse 3D points into a dense grid structure where each
cell contains the maximum height value detected at that location. This 2.5D representation
is more suitable for machine learning models and provides a compressed view of the 3D scene.

Key features:
- Reads PLY point cloud files using Open3D
- Applies spatial filtering to focus on relevant areas
- Creates height-based grid maps with configurable resolution
- Uses maximum height aggregation for multiple points per cell
- Handles boundary conditions and edge cases
- Outputs grid coordinates and heights as CSV files

The grid conversion process:
1. Load 3D point cloud from PLY file
2. Filter points within spatial boundaries
3. Convert world coordinates to grid indices
4. Aggregate heights using maximum value per cell
5. Extract non-floor positions and save as CSV
"""

# Standard library imports
import sys
import os
from multiprocessing import Pool

# Third-party imports
import open3d as o3d
import numpy as np

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants
from config import *

def process_file(i, file, ply_directory, folder_path):
    """
    Convert a single PLY point cloud file to 2.5D grid representation.
    
    This function processes one point cloud file by loading the 3D points,
    filtering them spatially, and converting them to a 2.5D height grid.
    The grid represents the maximum height detected at each spatial location.
    
    Args:
        i (int): File index for sequential naming
        file (str): Name of the PLY file to process
        ply_directory (str): Directory containing the PLY file
        folder_path (str): Output directory for the converted CSV file
    
    Returns:
        None: Creates a CSV file with grid coordinates and heights
        
    Process:
        1. Load point cloud using Open3D
        2. Extract point coordinates as numpy array
        3. Filter points within spatial boundaries
        4. Convert coordinates to grid indices
        5. Aggregate heights using maximum values
        6. Handle boundary conditions
        7. Extract non-floor positions
        8. Save as CSV with format: column, row, height
    """
    # Load the point cloud file using Open3D
    ply_path = os.path.join(ply_directory, file)
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)

    # Set up grid parameters from configuration
    grid_resolution = GRID_RESOLUTION
    x_min = (DIMENSION_X*GRID_RESOLUTION)/2
    y_min = (DIMENSION_Y*GRID_RESOLUTION)/2
    x_range = DIMENSION_X
    y_range = DIMENSION_Y

    # Initialize the 2.5D grid map with floor height values
    grid_map = np.full((y_range, x_range), NO_DATA_VALUE, dtype=float)

    # Filter points to keep only those within the specified spatial boundaries
    valid_points = points[
        (points[:, 0] > -x_min) & (points[:, 0] < x_min) &     # X-axis boundaries
        (points[:, 1] > -y_min) & (points[:, 1] < y_min)      # Y-axis boundaries
    ]

    # Convert world coordinates to grid indices
    x_indices = ((valid_points[:, 0] - x_min) / grid_resolution).astype(int)
    y_indices = ((valid_points[:, 1] - y_min) / grid_resolution).astype(int)
    
    # Normalize height values by grid resolution for consistent scaling
    z_values = valid_points[:, 2] / grid_resolution

    # Aggregate multiple points per cell using maximum height
    np.maximum.at(grid_map, (y_indices, x_indices), z_values)

    # Handle boundary conditions: set first row and column to floor height
    grid_map[0, :] = NO_DATA_VALUE  # First row (y=0)
    grid_map[:, 0] = NO_DATA_VALUE  # First column (x=0)

    # Extract positions where height exceeds floor level
    non_zero_indices = np.nonzero(grid_map != NO_DATA_VALUE)
    values = grid_map[non_zero_indices]
    
    # Create output array with format: [column, row, height]
    # Note: indices are swapped to match expected coordinate convention
    positions = np.column_stack((non_zero_indices[1], non_zero_indices[0], values))

    # Generate output filename with sequential index
    filename = f"{file[:-4]}_{i}.csv"
    file_path = os.path.join(folder_path, filename)
    
    # Save the grid data as CSV file
    # Format: column,row,height with specified precision
    np.savetxt(file_path, positions, delimiter=",", fmt=['%d', '%d', '%.2f'])

def generate_grid_map(ply_directory, folder_path, lidar_number):
    """
    Generate 2.5D grid maps for all PLY files from a specific LiDAR sensor.
    
    This function coordinates the conversion process for an entire sensor's dataset
    by processing all PLY files in parallel. It replaces placeholder 'X' in paths
    with the actual sensor number and ensures the output directory exists.
    
    Args:
        ply_directory (str): Template path to PLY files directory (contains 'X' placeholder)
        folder_path (str): Template path for output directory (contains 'X' placeholder)
        lidar_number (int): Number of the LiDAR sensor (1-based indexing)
    
    Returns:
        None
        
    Process:
        1. Replace path placeholders with sensor number
        2. Get sorted list of all PLY files
        3. Create output directory if needed
        4. Use multiprocessing to convert files in parallel
        5. Each worker processes one PLY file independently
    
    Performance:
        - Uses all available CPU cores for parallel processing
        - Files are processed independently for maximum efficiency
        - Sequential file indexing ensures consistent output naming
    """
    # Replace 'X' placeholder in paths with the actual LiDAR sensor number
    ply_directory = ply_directory.replace('X', str(lidar_number))
    folder_path = folder_path.replace('X', str(lidar_number))

    # Get sorted list of all PLY files in the source directory
    ply_files = sorted([f for f in os.listdir(ply_directory) if f.endswith('.ply')])

    # Create output directory if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Use multiprocessing to convert files in parallel for improved performance
    with Pool() as pool:
        pool.starmap(process_file, [(i, file, ply_directory, folder_path) for i, file in enumerate(ply_files)])

# Main execution block
if __name__ == "__main__":

    # Process each LiDAR sensor sequentially
    for i in range(NUMBER_OF_SENSORS):
        generate_grid_map(LIDAR_DIRECTORY, HEIGHTMAP_DIRECTORY, i+1)
        
        print("lidar" + str(i+1) + " completed")