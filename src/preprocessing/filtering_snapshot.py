"""
Bounding Box Visibility Filter Module

This module filters bounding boxes based on their point cloud visibility. It analyzes 
each bounding box to determine if it contains enough LiDAR points to be considered 
"visible" and marks them accordingly. Bounding boxes with insufficient points are 
flagged as not visible, which helps improve the quality of training data by removing 
occluded or barely visible objects.

The filtering is based on different thresholds for different object types:
- Vehicles require more points due to their larger size
- Bicycles have moderate requirements
- Pedestrians have the lowest threshold due to their smaller profile
"""

# Standard library imports
import sys
import os
import csv
import ast
import json
from multiprocessing import Pool

# Third-party imports
import numpy as np

# Add the parent directory to the Python path to access config and visualization modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants and visualization functions
from config import *
from viz import *

def get_min_height_from_json(lidar_number):
    """
    Compute the per-sensor min-height threshold from the sensor JSON.

    Reads SENSORS_DIR/lidar{lidar_number}.json, takes actors[0].location.z,
    divides it by GRID_RESOLUTION, and inverts the sign (same rule as offsets).

    Args:
        lidar_number (int): 1-based sensor index.

    Returns:
        float: Per-sensor minimum height (grid units).
    """
    cfg_path = os.path.join(SENSORS_DIR, f"lidar{lidar_number}.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        z = float(data["actors"][0]["location"]["z"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid JSON structure in {cfg_path}: {e}")

    if GRID_RESOLUTION == 0:
        raise ValueError("GRID_RESOLUTION must be non-zero.")

    # Same rule "like before": divide by grid resolution, invert sign
    min_height = -(z / GRID_RESOLUTION)
    return min_height


def eliminate_lines_from_file(BB_path, lines_to_eliminate):
    """
    Mark bounding boxes as visible or not visible in the CSV file.
    
    This function modifies the bounding box CSV file by adding a 'visible' column
    that indicates whether each bounding box contains sufficient LiDAR points.
    Lines specified in lines_to_eliminate are marked as 'no', others as 'yes'.
    
    Args:
        BB_path (str): Path to the bounding box CSV file
        lines_to_eliminate (list): List of line indices (0-based) to mark as not visible
    
    Returns:
        None: Modifies the file in-place
        
    Side Effects:
        - Overwrites the original CSV file with added visibility column
        - Marks each data row with visibility status
    """
    # Read all lines from the bounding box CSV file
    with open(BB_path, 'r') as file_BB:
        lines = file_BB.readlines()

    # Add 'visible' tag to the header row
    header = lines[0].strip() + ',visible\n'

    # Process each data line (skip header at index 0)
    modified_lines = []
    for idx, line in enumerate(lines[1:]):
        line = line.strip()
        # Mark as 0 if this line index is in the elimination list, otherwise 1
        visibility = 0 if idx in lines_to_eliminate else 1
        modified_lines.append(f"{line},{visibility}\n")

    # Write the modified content back to the file
    with open(BB_path, 'w') as file_BB:
        file_BB.write(header)
        file_BB.writelines(modified_lines)

def load_bounding_box(csv_file):
    """
    Load bounding box vertices and labels from a CSV file.
    
    This function reads a CSV file containing bounding box data and extracts
    the vertex information and object labels. The vertices are expected to be
    in string format in the third column and need to be parsed.
    
    Args:
        csv_file (str): Path to the CSV file containing bounding box data
    
    Returns:
        tuple: A tuple containing:
            - bounding_boxes (np.array): Array of bounding box vertices 
            - labels (list): List of object labels corresponding to each bounding box
            
    Note:
        The CSV format is expected to have:
        - Column 0: ID
        - Column 1: Object label (car, bicycle, pedestrian, etc.)
        - Column 2: Vertices as string representation of coordinates
    """
    bounding_box_vertices = []  # List to store vertex data
    labels = []  # List to store object labels

    # Read the CSV file and extract vertex and label information
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        
        for row in reader:
            # Extract vertices from the third column (index 2)
            vertices = [row[2]]
            bounding_box_vertices.append(vertices)
            
            # Extract label from the second column (index 1)
            labels.append(row[1])
    
    # Convert the list to numpy array for easier processing
    bounding_boxes = np.array(bounding_box_vertices)
    return bounding_boxes, labels

def process_file(args):
    """
    Process a single pair of LiDAR and bounding box files to determine visibility.

    ...
        args (tuple): Tuple containing:
            - path_lidar (str): Directory path containing LiDAR grid files
            - path_BB (str): Directory path containing bounding box files
            - lidar_file (str): Name of the LiDAR grid file to process
            - BB_file (str): Name of the bounding box file to process
            - lidar_number (int): 1-based sensor index  <-- NEW
    """
    # Unpack the argument tuple
    path_lidar, path_BB, lidar_file, BB_file, lidar_number = args  # <-- UPDATED

    # Construct the complete path to the LiDAR file
    complete_path_lidar = os.path.join(path_lidar, lidar_file)

    # Load the LiDAR data points from the grid file
    points = load_points_grid_map(complete_path_lidar)

    # Create a 2D height grid map initialized with floor height (unchanged)
    grid_map_recreate = np.full((DIMENSION_Y, DIMENSION_X), NO_DATA_VALUE, dtype=float)

    # Populate the grid map with height values
    for pos in points:
        col, row, height = pos
        grid_map_recreate[int(row), int(col)] = height

    # Load bounding box definitions and labels
    BB_path = os.path.join(path_BB, BB_file)
    bounding_box_vertices, labels = load_bounding_box(BB_path)

    # Indices of bounding boxes that don't have enough points
    bounding_boxes_without_points = []

    # Ground noise offset (~0.3m in grid units)
    height_offset = int(0.3 / GRID_RESOLUTION)

    min_height = get_min_height_from_json(lidar_number)

    # Analyze each bounding box
    for i, bb in enumerate(bounding_box_vertices):
        count = 0
        string_data = bb[0]
        pairs = ast.literal_eval(string_data)

        # Count points above threshold within the box
        for pair in pairs:
            col, row = pair
            if grid_map_recreate[int(row), int(col)] > (min_height + height_offset): 
                count += 1

        # Type-specific thresholds
        if labels[i] == "car":
            if count < MINIMUM_POINTS_VEHICLE:
                bounding_boxes_without_points.append(i)
        elif labels[i] == "bicycle":
            if count < MINIMUM_POINTS_BICYCLE:
                bounding_boxes_without_points.append(i)
        else:
            if count < MINIMUM_POINTS_PEDESTRIAN:
                bounding_boxes_without_points.append(i)

    # Mark not-visible boxes
    eliminate_lines_from_file(BB_path, bounding_boxes_without_points)


def eliminate_BB(path_lidar, path_BB, lidar_number):
    """
    Process all file pairs for a specific LiDAR sensor to filter bounding boxes.
    
    This function coordinates the visibility analysis for an entire sensor's data
    by processing all corresponding LiDAR and bounding box file pairs in parallel.
    It replaces placeholder 'X' in paths with the actual sensor number and uses
    multiprocessing to efficiently handle large datasets.
    
    Args:
        path_lidar (str): Template path to LiDAR grid directory (contains 'X' placeholder)
        path_BB (str): Template path to bounding box directory (contains 'X' placeholder) 
        lidar_number (int): Number of the LiDAR sensor (1-based indexing)
    
    Returns:
        None
    """
    # Replace 'X' placeholder in paths with the actual LiDAR sensor number
    path_lidar = path_lidar.replace('X', str(lidar_number))
    path_BB = path_BB.replace('X', str(lidar_number))

    # Get sorted lists of CSV files from both directories
    # Sorting ensures consistent pairing between LiDAR and bounding box files
    lidar_files = sorted([f for f in os.listdir(path_lidar) if f.endswith('.csv')])
    BB_files = sorted([f for f in os.listdir(path_BB) if f.endswith('.csv')])

    # Use multiprocessing to process file pairs in parallel for improved performance
    # Each worker process handles one (LiDAR file, bounding box file) pair
    with Pool() as pool:
        pool.map(
            process_file,
            [(path_lidar, path_BB, lidar_file, BB_file, lidar_number)
                for lidar_file, BB_file in zip(lidar_files, BB_files)]
        )

# Main execution block
if __name__ == "__main__":
    
    # Process each LiDAR sensor sequentially
    for i in range(NUMBER_OF_SENSORS):
        # Process sensor number (i+1) since sensors are 1-indexed
        # HEIGHTMAP_DIRECTORY: source directory for LiDAR grid data
        # OGM_DIRECTORY: directory containing bounding box data to filter
        eliminate_BB(HEIGHTMAP_DIRECTORY, OGM_DIRECTORY, i+1)
        
        # Provide progress feedback to user
        print("lidar" + str(i+1) + " completed")