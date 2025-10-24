import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import open3d as o3d
import csv
import numpy as np
import math

def rotate_point(point, rotation_matrix):
    """
    Apply 3D rotation transformation to a single point using rotation matrix.
    
    This function performs 3D coordinate transformation by multiplying a point's
    coordinates with a 3x3 rotation matrix. Used for rotating bounding box vertices
    to match their orientation in 3D space before visualization.
    
    Args:
        point (tuple/list): 3D point coordinates as (x, y, z)
                           Represents a vertex or spatial position to be rotated
        rotation_matrix (list): 3x3 rotation matrix as nested list
                               [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
                               Defines the rotation transformation to apply
    
    Returns:
        tuple: Rotated 3D point coordinates as (new_x, new_y, new_z)
               Result of matrix multiplication: rotation_matrix * point
    """
    x, y, z = point  # Extract individual coordinates from input point
    return (
        # Apply first row of rotation matrix: new_x = r00*x + r01*y + r02*z
        rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2] * z,
        # Apply second row of rotation matrix: new_y = r10*x + r11*y + r12*z
        rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2] * z,
        # Apply third row of rotation matrix: new_z = r20*x + r21*y + r22*z
        rotation_matrix[2][0] * x + rotation_matrix[2][1] * y + rotation_matrix[2][2] * z
    )

def get_rotation_matrix(pitch, roll, yaw):
    """
    Generate 3D rotation matrix from Euler angles for bounding box orientation.
    
    IMPORTANT COMPATIBILITY NOTE:
    This function maintains compatibility with the original V2X dataset visualizer
    by reordering input angles and applying the exact rotation matrix formula used
    in the reference implementation. Input angles are provided as (pitch, roll, yaw)
    but internally reordered to (roll, yaw, pitch) to match visualizer convention.
    
    Args:
        pitch (float): Pitch angle in degrees (rotation around X-axis)
                      Positive values indicate nose-up rotation
        roll (float): Roll angle in degrees (rotation around Y-axis)  
                     Positive values indicate roll to the right
        yaw (float): Yaw angle in degrees (rotation around Z-axis)
                    Positive values indicate counterclockwise rotation (top view)
    
    Returns:
        list: 3x3 rotation matrix as nested list format
              [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
              Ready for use with rotate_point() function
    
    Coordinate System:
        - X-axis: Forward (pitch rotation)
        - Y-axis: Right (roll rotation)
        - Z-axis: Up (yaw rotation)
        - Right-handed coordinate system
    """
    # Reorder angle parameters to match original V2X visualizer convention
    roll_deg = roll    # Store roll angle (Y-axis rotation)
    yaw_deg = yaw      # Store yaw angle (Z-axis rotation)  
    pitch_deg = pitch  # Store pitch angle (X-axis rotation)

    # Convert angles from degrees to radians for trigonometric functions
    r = math.radians(roll_deg)   # Roll angle in radians
    y = math.radians(yaw_deg)    # Yaw angle in radians
    p = math.radians(pitch_deg)  # Pitch angle in radians

    # Pre-compute trigonometric values for efficiency
    c_y, s_y = math.cos(y), math.sin(y)  # Cosine and sine of yaw
    c_r, s_r = math.cos(r), math.sin(r)  # Cosine and sine of roll
    c_p, s_p = math.cos(p), math.sin(p)  # Cosine and sine of pitch

    # Construct 3x3 rotation matrix using V2X visualizer formula
    return [
        # First row: [cp*cy, cy*sp*sr - sy*cr, -cy*sp*cr - sy*sr]
        [ c_p * c_y,               c_y * s_p * s_r - s_y * c_r,   -c_y * s_p * c_r - s_y * s_r ],
        # Second row: [sy*cp, sy*sp*sr + cy*cr, -sy*sp*cr + cy*sr] 
        [ s_y * c_p,               s_y * s_p * s_r + c_y * c_r,   -s_y * s_p * c_r + c_y * s_r ],
        # Third row: [sp, -cp*sr, cp*cr]
        [ s_p,                    -c_p * s_r,                      c_p * c_r                   ],
    ]

def load_bounding_box(csv_file):
    """
    Load and construct 3D bounding boxes from CSV detection data.
    
    This function reads object detection results from CSV files and constructs
    the 8 corner vertices for each 3D bounding box. It processes object center
    coordinates, dimensions, and orientation angles to create accurate 3D box
    representations for visualization overlay on point cloud data.
    
    Args:
        csv_file (str): Path to CSV file containing bounding box detection data
                       Expected format: header + rows with object information
    
    Returns:
        np.ndarray: Array of bounding box vertex sets with shape (num_objects, 8, 3)
                   Each object has 8 vertices representing box corners in 3D space
    
    CSV Data Format:
        - Column 0: Object ID
        - Column 1: Object class (e.g., 'pedestrians', 'vehicles') 
        - Columns 2-4: Object center coordinates (x, y, z)
        - Columns 5-7: Object dimensions (length, width, height)
        - Columns 8-10: Orientation angles (pitch, roll, yaw) in degrees
    """
    bounding_box_vertices = []  # Initialize list to store all bounding box vertices

    # Open CSV file and process each object detection row
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row to access object data
        
        # Process each detected object in the CSV file
        for row in reader:
            # Apply size increment for pedestrian objects (safety margin)
            if row[1].strip().lower() == 'pedestrians':
                increment = INCREMENT_BB_PEDESTIAN  # Add extra space around pedestrians
            else:
                increment = 0.0  # No size adjustment for other object types

            # Extract object properties from CSV row
            center = [float(row[2]), float(row[3]), float(row[4])]      # Object center (x, y, z)
            dimension = [float(row[5]) + increment, float(row[6]) + increment, float(row[7]) + increment]  # Dimensions with increment
            rotation = [float(row[8]), float(row[9]), float(row[10])]  # Euler angles (pitch, roll, yaw)

            # Define 8 corner offsets for a rectangular bounding box (relative to center)
            offsets = [
                # Top face corners (positive Z)
                [ dimension[0],  dimension[1],  dimension[2] ],   # Top-front-right
                [ dimension[0], -dimension[1],  dimension[2] ],   # Top-front-left
                [-dimension[0],  dimension[1],  dimension[2] ],   # Top-back-right
                [-dimension[0], -dimension[1],  dimension[2] ],   # Top-back-left
                # Bottom face corners (negative Z)
                [ dimension[0],  dimension[1], -dimension[2] ],   # Bottom-front-right
                [ dimension[0], -dimension[1], -dimension[2] ],   # Bottom-front-left
                [-dimension[0],  dimension[1], -dimension[2] ],   # Bottom-back-right
                [-dimension[0], -dimension[1], -dimension[2] ]    # Bottom-back-left
            ]

            # Build rotation matrix using the original visualizer's convention
            rotation_matrix = get_rotation_matrix(rotation[0], rotation[1], rotation[2])

            # Transform each corner offset: rotate then translate to world coordinates
            vertices = []
            for offset in offsets:
                # Apply rotation transformation to corner offset
                rx, ry, rz = rotate_point(offset, rotation_matrix)
                # Translate rotated point to world position by adding object center
                vertices.append((center[0] + rx, center[1] + ry, center[2] + rz))

            # Add this object's 8 vertices to the collection
            bounding_box_vertices.append(vertices)

    # Convert to numpy array for efficient processing and return
    bounding_boxes = np.array(bounding_box_vertices)
    return bounding_boxes

def create_bounding_box_lines(vertices):
    """
    Create Open3D LineSet for wireframe visualization of 3D bounding boxes.
    
    This function constructs a wireframe representation of a 3D bounding box
    by connecting the 8 corner vertices with 12 line segments. The resulting
    LineSet can be added to Open3D visualizer to display object boundaries
    as colored wireframes overlaid on point cloud data.
    
    Args:
        vertices (array-like): 8 vertices of the bounding box
                              Shape: (8, 3) where each vertex is [x, y, z]
                              Vertices represent the 8 corners of a rectangular box
    
    Returns:
        o3d.geometry.LineSet: Open3D LineSet object for 3D wireframe visualization
                             Contains all 12 edges of the rectangular bounding box
    """
    # Define connectivity for 12 edges of a rectangular bounding box
    lines = [
        # Bottom face edges (4 lines forming bottom rectangle)
        [0, 1], [1, 3], [3, 2], [2, 0],  # Connect bottom corners in rectangular loop
        # Top face edges (4 lines forming top rectangle)  
        [4, 5], [5, 7], [7, 6], [6, 4],  # Connect top corners in rectangular loop
        # Vertical edges (4 lines connecting bottom to top)
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connect corresponding bottom-top vertex pairs
    ]

    # Create Open3D LineSet object for wireframe visualization
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)  # Set the 8 corner vertices
    line_set.lines = o3d.utility.Vector2iVector(lines)      # Set the 12 connecting lines
    return line_set

def visualize_ply_with_bounding_boxes(ply_directory, csv_directory, specific_ply):
    """
    Display interactive 3D visualization of LiDAR point clouds with object bounding boxes.
    
    This function provides comprehensive visualization of LiDAR detection results by
    combining point cloud data with 3D bounding box overlays. It processes paired
    .ply and .csv files to create interactive visualizations for detection validation,
    algorithm evaluation, and spatial data analysis.
    
    Args:
        ply_directory (str): Path to directory containing .ply point cloud files
                            Files should be sequentially numbered/ordered for proper pairing
        csv_directory (str): Path to directory containing .csv bounding box files  
                            Must have matching filenames with corresponding .ply files
        specific_ply (int): Starting index for file processing
                           If >= 0 and < number_files: starts from specified index
                           If invalid: displays error and processes all files
    
    Returns:
        None: Function displays interactive visualization windows for each file pair
    """
    # Load and sort files from both directories for proper pairing
    ply_files = sorted([f for f in os.listdir(ply_directory) if f.endswith('.ply')])  # Point cloud files
    csv_files = sorted([f for f in os.listdir(csv_directory) if f.endswith('.csv')])  # Bounding box files

    # Validate that file counts match between directories
    if len(ply_files) != len(csv_files):
        print(f"Warning: The number of .ply files ({len(ply_files)}) and .csv files ({len(csv_files)}) do not match!")
        return
    
    # Apply index filtering based on specific_ply parameter
    if (specific_ply >= 0 and specific_ply < len(ply_files)):
        # Start processing from specified index to end
        ply_files = ply_files[specific_ply:]
        csv_files = csv_files[specific_ply:]
    else:
        print(f"ERROR : {specific_ply} is not correct")  # Invalid index, process all files

    # Process each file pair sequentially for visualization
    for i, ply_file in enumerate(ply_files):
        # Create full-screen Open3D visualizer window
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=1920, height=1200, left=0, top=80, visible=True)

        # Load point cloud data from .ply file
        ply_path = os.path.join(ply_directory, ply_file)
        cloud = o3d.io.read_point_cloud(ply_path)  # Read LiDAR point cloud

        # Load corresponding bounding box data from .csv file
        csv_file = csv_files[i] 
        csv_path = os.path.join(csv_directory, csv_file)
        bounding_box_vertices = load_bounding_box(csv_path)  # Parse detection results

        # Add wireframe bounding boxes to visualization
        for vertices in bounding_box_vertices:
            box_lines = create_bounding_box_lines(vertices)  # Create wireframe for each object
            vis.add_geometry(box_lines)  # Add bounding box to visualizer

        # Add point cloud to visualization
        vis.add_geometry(cloud)  # Add LiDAR data to visualizer

        # Update and launch interactive visualization window
        vis.update_geometry(cloud)   # Refresh geometry rendering
        vis.poll_events()            # Process window events
        vis.update_renderer()        # Update display rendering
        vis.run()                    # Launch interactive window (blocks until closed)
        vis.destroy_window()         # Clean up resources after window closes

if __name__ == "__main__":
    import sys
    
    # Check if arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python plot_pointcloud.py <lidar_number> <starting_file_number>")
        print("Example: python plot_pointcloud.py 1 50")
        sys.exit(1)
    
    try:
        lidar_number = int(sys.argv[1])
        starting_file = int(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be integers")
        sys.exit(1)
    
    # Replace 'X' placeholder in directory paths with selected sensor number
    path_lidar = LIDAR_DIRECTORY.replace('X', str(lidar_number))        # Point cloud directory path
    new_position_path = SNAPSHOT_SYNCRONIZED_DIRECTORY.replace('X', str(lidar_number))  # Bounding box directory path

    # Launch visualization with selected sensor parameters
    visualize_ply_with_bounding_boxes(path_lidar, new_position_path, starting_file)
