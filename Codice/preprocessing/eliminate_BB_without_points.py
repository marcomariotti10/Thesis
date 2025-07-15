import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from visualization import *
import os
import numpy as np
import csv
import ast
from multiprocessing import Pool

def eliminate_lines_from_file(BB_path, lines_to_eliminate):
    with open(BB_path, 'r') as file_BB:
        lines = file_BB.readlines()

    header = lines[0].strip() + ',visible\n'  # Add 'visible' tag to header

    modified_lines = []
    for idx, line in enumerate(lines[1:]):
        line = line.strip()
        visibility = 'no' if idx in lines_to_eliminate else 'yes'
        modified_lines.append(f"{line},{visibility}\n")

    with open(BB_path, 'w') as file_BB:
        file_BB.write(header)
        file_BB.writelines(modified_lines)

def load_bounding_box(csv_file):
    """Load bounding box vertices from a CSV file."""
    min_height = MIN_HEIGHT
    bounding_box_vertices = []
    labels = [] 

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            vertices = [row[2]]
            bounding_box_vertices.append(vertices)
            labels.append(row[1])
    bounding_boxes = np.array(bounding_box_vertices)
    return bounding_boxes, labels

def process_file(args):
    path_lidar, path_BB, lidar_file, BB_file = args

    complete_path_lidar = os.path.join(path_lidar, lidar_file)

    # Load the lidar data points
    points = load_points_grid_map(complete_path_lidar)

    # Recreate the grid map from positions array
    grid_map_recreate = np.full((Y_RANGE, X_RANGE), FLOOR_HEIGHT, dtype=float)

    # Fill the grid map with values from positions array
    for pos in points:
        col, row, height = pos
        grid_map_recreate[int(row), int(col)] = height

    # Get the points of BB
    BB_path = os.path.join(path_BB, BB_file)
    bounding_box_vertices, labels = load_bounding_box(BB_path)

    # Create a list to store the bounding boxes with points
    bounding_boxes_without_points = []

    # Loop through each bounding box and select the ones with too few points
    for i, bb in enumerate(bounding_box_vertices):
        count = 0

        string_data = bb[0]
        # Safely evaluate the string to convert it into a list of tuples
        pairs = ast.literal_eval(string_data)
        
        for pair in pairs:
            col, row = pair
            if grid_map_recreate[int(row), int(col)] > MIN_HEIGHT + HEIGHT_OFFSET:
                count += 1 
        
        if (labels[i] == "car"):
            if count < NUM_MIN_POINTS_VEHICLE:
                bounding_boxes_without_points.append(i)
        elif (labels[i] == "bicycle"): 
            if count < NUM_MIN_POINTS_BICYCLE:
                bounding_boxes_without_points.append(i)
        else:
            if count < NUM_MIN_POINTS_PEDESTRIAN:
                bounding_boxes_without_points.append(i)
    
    eliminate_lines_from_file(BB_path, bounding_boxes_without_points)

def eliminate_BB(path_lidar, path_BB, lidar_number):
    
    # Replace 'X' in the paths with the lidar_number
    path_lidar = path_lidar.replace('X', str(lidar_number))
    path_BB = path_BB.replace('X', str(lidar_number))

    # Get the list of files in the specified folder
    lidar_files = sorted([f for f in os.listdir(path_lidar) if f.endswith('.csv')])
    BB_files = sorted([f for f in os.listdir(path_BB) if f.endswith('.csv')])

    # Use multiprocessing to process files in parallel
    with Pool() as pool:
        pool.map(process_file, [(path_lidar, path_BB, lidar_file, BB_file) for lidar_file, BB_file in zip(lidar_files, BB_files)])

if __name__ == "__main__":

    while True:
        user_input = input("Enter the number of the lidar for the single lidar, or enter 'all' to process all the lidar: ")
        if user_input == 'all':
            for i in range(NUMBER_OF_SENSORS):
                eliminate_BB(LIDAR_X_GRID_DIRECTORY, SNAPSHOT_X_GRID_DIRECTORY, i+1)
                print("lidar" + str(i+1) + " done")
            break
        elif (int(user_input) in range(1, NUMBER_OF_SENSORS+1)):
            eliminate_BB(LIDAR_X_GRID_DIRECTORY, SNAPSHOT_X_GRID_DIRECTORY, int(user_input))
            break
        else:
            print("Invalid input.")