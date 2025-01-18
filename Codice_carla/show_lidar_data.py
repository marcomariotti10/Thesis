import datetime
import sys
import os
import shutil
import open3d as o3d
import csv
import numpy as np
from constants import *


def load_bounding_box(csv_file):
    """Load bounding box vertices from a CSV file."""
    bounding_box_vertices = []

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            vertices = [
                [float(row[i]), float(row[i + 1]), float(row[i + 2])] for i in range(5, 29, 3)
            ]
            bounding_box_vertices.append(vertices)
    bounding_boxes = np.array(bounding_box_vertices)
    print(vertices)
    print(bounding_box_vertices)
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
    ply_files = [f for f in os.listdir(ply_directory) if f.endswith('.ply')]
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    print(specific_ply)

    if len(ply_files) != len(csv_files):
        print(f"Warning: The number of .ply files ({len(ply_files)}) and .csv files ({len(csv_files)}) do not match!")
        return
    try:
        index = ply_files.index(specific_ply)
        print(f"The index of {specific_ply} is: {index}")
        ply_files = ply_files[index:]
        csv_files = csv_files[index:]
    except ValueError:
        print(f"ERROR : {specific_ply} is not in the list")
        sys.exit(1)
    
    for i, ply_file in enumerate(ply_files) : 
        # Create Open3D Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Open3D', width=1920, height=1200, left=0, top=30, visible=True)

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

        # Set the viewpoint (e.g., camera parameters)
        view_control = vis.get_view_control()
        view_control.set_front([0.0, 0.0, -1.0])
        view_control.set_lookat([0.0, 0.0, 0.0])
        view_control.set_up([0.0, -1.0, 0.0])
        view_control.set_zoom(0.8)

        # Run the visualizer
        vis.run()

        # Close the visualizer after the user closes the window
        vis.destroy_window()

if __name__ == "__main__":

    path_lidar_output_cross_s = LIDAR_DIRECTORY
    new_path_position = NEW_POSITION_LIDAR_CROSS_S_DIRECTORY

    visualize_ply_with_bounding_boxes(path_lidar_output_cross_s, new_path_position, LIDAR_FILE)


