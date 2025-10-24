"""
Bounding Box to 2.5D Grid Conversion Module

This module converts 3D bounding box definitions into 2.5D grid representations.
It processes CSV files containing bounding box parameters (center, dimensions, rotation)
and transforms them into grid-based footprints that can be used for training and
evaluation in machine learning models.

Key features:
- Applies 3D rotations using proper rotation matrices
- Converts 3D bounding boxes to 2.5D grid footprints
- Filters boxes based on spatial boundaries
- Handles different object types with appropriate size adjustments
- Uses polygon filling for accurate grid representation

The conversion process involves:
1. Reading bounding box parameters from CSV
2. Generating 3D corner vertices
3. Applying rotational transformations
4. Projecting to 2.5D grid space
5. Filling polygon areas on the grid
"""

# Standard library imports
import sys
import os
import csv
import math
from multiprocessing import Pool

# Third-party imports
import numpy as np

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants
from config import *

def rotate_point(point, rotation_matrix):
    """
    Apply a 3D rotation matrix to a single point.
    
    This function performs matrix multiplication to rotate a 3D point using
    the provided rotation matrix. It's used to transform bounding box corners
    from their local coordinate system to the world coordinate system.
    
    Args:
        point (tuple/list): 3D point coordinates (x, y, z)
        rotation_matrix (list): 3x3 rotation matrix as nested lists
    
    Returns:
        tuple: Rotated 3D point coordinates (x', y', z')
        
    Mathematical operation:
        [x']   [R00 R01 R02] [x]
        [y'] = [R10 R11 R12] [y]
        [z']   [R20 R21 R22] [z]
    """
    x, y, z = point
    return (
        rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z,
        rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z,
        rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z
    )

def fill_polygon(grid_map, vertices, height):
    # Create an empty mask with the same shape as the grid map
    mask = np.zeros_like(grid_map, dtype=np.uint8)
    
    # Convert vertices to integer coordinates
    vertices_int = np.array(vertices[:, :2], dtype=np.int32)
    #print("vertices_int", vertices_int)
    
    # Define different orders to try
    orders = [
        [0, 1, 3, 2],
        [0, 1, 2, 3]
    ]
    
    # Try filling the polygon with different orders of vertices
    for order in orders:
        ordered_vertices = vertices_int[order]
        cv2.fillPoly(mask, [ordered_vertices], 1)
    
    # Set the height for the filled area in the grid map
    grid_map[mask == 1] = height

def get_rotation_matrix(pitch, roll, yaw):
    """
    Generate a 3D rotation matrix from Euler angles.
    
    This function creates a rotation matrix that matches the V2X visualizer convention.
    The CSV provides angles as (pitch, roll, yaw), but the rotation matrix is built
    using the order (roll, yaw, pitch) to match the original visualization system.
    
    Args:
        pitch (float): Rotation around X-axis in degrees  
        roll (float): Rotation around Y-axis in degrees
        yaw (float): Rotation around Z-axis in degrees
    
    Returns:
        list: 3x3 rotation matrix as nested lists
        
    Mathematical Formula:
        The rotation matrix is computed as R = Rz(yaw) * Ry(roll) * Rx(pitch)
        Result matrix:
        [ cp*cy,           cy*sp*sr - sy*cr,   -cy*sp*cr - sy*sr ]
        [ sy*cp,           sy*sp*sr + cy*cr,   -sy*sp*cr + cy*sr ]
        [ sp,             -cp*sr,              cp*cr             ]
        
        Where: c = cos, s = sin, p = pitch, r = roll, y = yaw
    """
    # Reorder angles to match V2X visualizer convention
    roll_deg = roll
    yaw_deg = yaw  
    pitch_deg = pitch

    # Convert degrees to radians for trigonometric functions
    r = math.radians(roll_deg)
    y = math.radians(yaw_deg)
    p = math.radians(pitch_deg)

    # Pre-compute trigonometric values for efficiency
    c_y, s_y = math.cos(y), math.sin(y)  # cosine and sine of yaw
    c_r, s_r = math.cos(r), math.sin(r)  # cosine and sine of roll
    c_p, s_p = math.cos(p), math.sin(p)  # cosine and sine of pitch

    # Construct the rotation matrix using the V2X visualizer formula
    return [
        [ c_p * c_y,               c_y * s_p * s_r - s_y * c_r,   -c_y * s_p * c_r - s_y * s_r ],
        [ s_y * c_p,               s_y * s_p * s_r + c_y * c_r,   -s_y * s_p * c_r + c_y * s_r ],
        [ s_p,                    -c_p * s_r,                      c_p * c_r                   ],
    ]

def process_csv_file(args):
    """
    Convert bounding boxes from 3D definitions to 2.5D grid representations.
    
    This function processes a single CSV file containing 3D bounding box definitions
    and converts them into 2.5D grid-based representations. The process involves:
    1. Reading bounding box parameters (center, dimensions, rotation)
    2. Generating 3D corner vertices for each bounding box
    3. Applying rotational transformations
    4. Filtering boxes based on spatial boundaries
    5. Converting to grid coordinates and filling polygons
    6. Extracting grid positions for each bounding box
    
    Args:
        args (tuple): Tuple containing:
            - BB_path (str): Directory path containing bounding box CSV files
            - file (str): Name of the CSV file to process
            - folder_path (str): Output directory for converted grid files
    
    Returns:
        None: Creates a new CSV file with grid-based bounding box representations
        
    CSV Input Format:
        Column 0: Actor ID
        Column 1: Object label (car, bicycle, pedestrian, etc.)
        Columns 2-4: Center coordinates (x, y, z)
        Columns 5-7: Dimensions (length, width, height)
        Columns 8-10: Rotation angles (pitch, roll, yaw)
        
    CSV Output Format:
        actor_id: Object identifier
        label: Object type
        points: List of grid coordinates occupied by the bounding box
    """
    # Unpack arguments
    BB_path, file, folder_path = args
    
    #How much increase the grid map during the conversion of bounding boxes from 3D to 2.5D (recommended half of DIMENSION_X)
    INCREASE_GRID_RANGE = DIMENSION_X // 2
    X_MIN = (DIMENSION_X*GRID_RESOLUTION)//2
    Y_MIN = (DIMENSION_Y*GRID_RESOLUTION)//2

    # Set up grid parameters from configuration
    grid_resolution = GRID_RESOLUTION
    x_min = X_MIN + (int(INCREASE_GRID_RANGE/2)*grid_resolution)
    y_min = Y_MIN + (int(INCREASE_GRID_RANGE/2)*grid_resolution)
    x_range = DIMENSION_X
    y_range = DIMENSION_Y

    # Open and read the bounding box CSV file
    reader = csv.reader(open(os.path.join(BB_path, file), encoding='utf-8'))

    # Initialize lists to store processed data
    bounding_box_vertices = []  # 3D vertices for each valid bounding box
    bounding_box_ids = []       # Actor IDs for each bounding box
    bounding_box_labels = []    # Object labels for each bounding box
    all_positions = []          # Final grid positions for each bounding box

    next(reader)  # Skip the header row

    # Process each bounding box definition in the CSV
    for row in reader:
        # Apply size increment for pedestrians (improves detection)
        if row[1].strip().lower() == 'pedestrian':
            increment = INCREMENT_BB_PEDESTIAN
        else:
            increment = 0.0

        # Extract bounding box parameters from CSV row
        center = [float(row[2]), float(row[3]), float(row[4])]  # Center coordinates
        dimension = [float(row[5]) + increment, float(row[6]) + increment, float(row[7]) + increment]  # Dimensions with increment
        rotation = [float(row[8]), float(row[9]), float(row[10])]  # Rotation angles (pitch, roll, yaw)

        # Define the four base corners of the bounding box in local coordinates
        offsets = [
            [ dimension[0],  dimension[1],  dimension[2] ],  # Front-right corner
            [ dimension[0], -dimension[1],  dimension[2] ],  # Front-left corner  
            [-dimension[0],  dimension[1],  dimension[2] ],  # Back-right corner
            [-dimension[0], -dimension[1],  dimension[2] ],  # Back-left corner
        ]

        # Build rotation matrix using the V2X visualizer convention
        rotation_matrix = get_rotation_matrix(rotation[0], rotation[1], rotation[2])

        # Transform each corner from local to world coordinates
        vertices = []
        for offset in offsets:
            # Apply rotation to the offset vector
            rx, ry, rz = rotate_point(offset, rotation_matrix)
            # Translate by the center position to get world coordinates
            vertices.append((center[0] + rx, center[1] + ry, center[2] + rz))

        # Quality control: filter out invalid bounding boxes
        x_old = 10000  # Track coordinate changes to detect degenerate boxes
        y_old = 10000
        z_old = 10000
        outside_count = 0      # Count vertices outside the valid range
        all_in_range = True    # Flag to track if bounding box is within bounds
        all_same = False       # Flag to detect degenerate (zero-area) boxes
        
        # Check each vertex for validity
        for ver in vertices:
            # Count vertices outside the spatial boundaries
            if ((ver[0] > X_MIN - 0.5) or (ver[0] < -X_MIN + 0.5) or 
                (ver[1] > Y_MIN - 0.5) or (ver[1] < -Y_MIN + 0.5)):
                outside_count += 1
                # If all vertices are outside, mark as invalid
                if outside_count >= 4:
                    all_in_range = False
                    break
            
            # Check for degenerate bounding boxes (all vertices at same position)
            if ((ver[0] != x_old) or (ver[1] != y_old) or (ver[2] != z_old)):
                x_old = ver[0]
                y_old = ver[1] 
                z_old = ver[2]
            else:
                all_same = True
                break

        # Only keep valid bounding boxes (within range and not degenerate)
        if all_in_range and not all_same:
            bounding_box_vertices.append(vertices)
            bounding_box_ids.append(row[0])
            bounding_box_labels.append(row[1])

    # Convert each valid bounding box to grid coordinates
    for vertic in bounding_box_vertices:
        # Create an extended grid map (larger than final output for boundary handling)
        grid_map = np.full((y_range + INCREASE_GRID_RANGE, x_range + INCREASE_GRID_RANGE), 0, dtype=float)
        grid_vertices = []
        
        # Convert world coordinates to grid indices and mark vertices
        for point in vertic:
            x, y, z = point
            x_idx = int((x - x_min) / grid_resolution)
            y_idx = int((y - y_min) / grid_resolution)
            grid_map[y_idx, x_idx] = z  # Store height at grid position
        
        # Find all non-zero grid positions (vertices)
        grid_vertices = np.nonzero(grid_map != 0)
        positions_array = np.column_stack((grid_vertices[1], grid_vertices[0]))
        positions = [tuple(row) for row in positions_array.tolist()]
        position = np.array(positions)
        
        # Fill the polygon defined by the vertices
        height_BB = 1  # Use constant height for polygon filling
        fill_polygon(grid_map, position, height_BB)

        # Extract the final grid region (remove the extended boundaries)
        increase_grid_range_half = int(INCREASE_GRID_RANGE / 2)
        smaller_grid = grid_map[(increase_grid_range_half):(increase_grid_range_half + y_range), 
                               (increase_grid_range_half):(increase_grid_range_half + x_range)]

        # Get all non-zero positions in the final grid (the filled polygon)
        non_zero_indices = np.nonzero(smaller_grid != 0)
        positions_array = np.column_stack((non_zero_indices[1], non_zero_indices[0]))
        positions = [tuple(row) for row in positions_array.tolist()]
        all_positions.append(positions)

    # Prepare data for output CSV
    data = [
        {
            "actor_id": actor_id,
            "label": label,
            "points": vert
        }
        for actor_id, label, vert in zip(bounding_box_ids, bounding_box_labels, all_positions)
    ]

    # Write the converted data to output CSV file
    path = os.path.join(folder_path, file)
    os.makedirs(folder_path, exist_ok=True)

    with open(path, 'w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'actor_id', 'label',
            'points'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def convert_BB_into_25D(BB_path, folder_path, lidar_number):
    """
    Convert all bounding box files for a specific LiDAR sensor to 2.5D grid format.
    
    This function coordinates the conversion process for an entire sensor's dataset
    by processing all CSV files in parallel. It replaces placeholder 'X' in paths
    with the actual sensor number and uses multiprocessing for efficiency.
    
    Args:
        BB_path (str): Template path to bounding box directory (contains 'X' placeholder)
        folder_path (str): Template path for output directory (contains 'X' placeholder)
        lidar_number (int): Number of the LiDAR sensor (1-based indexing)
    
    Returns:
        None
        
    Process:
        1. Replace path placeholders with sensor number
        2. Get list of all CSV files to process
        3. Use multiprocessing to convert files in parallel
        4. Each worker processes one CSV file independently
    """
    # Replace 'X' placeholder in paths with the actual LiDAR sensor number
    BB_path = BB_path.replace('X', str(lidar_number))
    folder_path = folder_path.replace('X', str(lidar_number))

    # Get sorted list of all CSV files in the source directory
    csv_files = sorted([f for f in os.listdir(BB_path) if f.endswith('.csv')])
    
    # Use multiprocessing to convert files in parallel for improved performance
    with Pool() as pool:
        pool.map(process_csv_file, [(BB_path, file, folder_path) for file in csv_files])

# Main execution block
if __name__ == "__main__":
    
    # Process each LiDAR sensor sequentially
    for i in range(NUMBER_OF_SENSORS):

        convert_BB_into_25D(SNAPSHOT_SYNCRONIZED_DIRECTORY, OGM_DIRECTORY, i+1)
        
        print("lidar" + str(i+1) + " completed")
