import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from visualization import *
import os
import numpy as np
import csv
from multiprocessing import Pool

def extract_smaller_grid(grid_map_recreate, positions, label):
    # Extract x and y coordinates from positions
    x_coords = [int(pos[0]) for pos in positions]
    y_coords = [int(pos[1]) for pos in positions]

    # Determine the bounding box 
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Extract the smaller grid, considering a margin for pedestrians because the BB is too small
    if(label == "vehicle" or label == "bicycle"):
        smaller_grid = grid_map_recreate[min_y:max_y, min_x:max_x]
    else:
        if min_y - 1 < 0:
                min_y = 0
        if min_x - 1 < 0:   
                min_x = 0
        if max_y + 1 > Y_RANGE:
                max_y = Y_RANGE
        if max_x + 1 > X_RANGE:
                max_x = X_RANGE
        smaller_grid = grid_map_recreate[min_y-INCREMENT_BB_PEDESTRIAN:max_y+INCREMENT_BB_PEDESTRIAN, min_x-INCREMENT_BB_PEDESTRIAN:max_x+INCREMENT_BB_PEDESTRIAN]

    return smaller_grid

def eliminate_lines_from_file(BB_path, new_path_output, file_name, lines_to_eliminate):
    with open(BB_path, 'r') as file_BB:
        lines = file_BB.readlines()

    header = lines[0]
    remaining_lines = [line for idx, line in enumerate(lines[1:]) if idx not in lines_to_eliminate]

    path = os.path.join(new_path_output, file_name)
    
    # Put the header and the remaining lines in the new file
    with open(path, 'w') as new_file:
        new_file.write(header)
        new_file.writelines(remaining_lines)

def load_bounding_box(csv_file):
    """Load bounding box vertices from a CSV file."""
    min_height = MIN_HEIGHT
    bounding_box_vertices = []
    labels = [] 

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            # Extract the 3D coordinates of the 8 bounding box vertices
            vertices = [
                [float(row[i]), float(row[i + 1]), float(row[i + 2])] for i in range(2, 12, 3)]
            bounding_box_vertices.append(vertices)
            labels.append(row[1])
    bounding_boxes = np.array(bounding_box_vertices)
    return bounding_boxes, labels

def process_file(args):
    path_lidar, path_BB, new_path_output, lidar_file, BB_file = args

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
        smaller_grid = extract_smaller_grid(grid_map_recreate, bb, labels[i])
        non_zero_indices = np.nonzero(smaller_grid > MIN_HEIGHT + HEIGHT_OFFSET)
        if (labels[i] == "car"):
            if len(non_zero_indices[0]) < NUM_MIN_POINTS_VEHICLE:
                bounding_boxes_without_points.append(i)
        elif (labels[i] == "bicycle"): 
            if len(non_zero_indices[0]) < NUM_MIN_POINTS_BICYCLE:
                bounding_boxes_without_points.append(i)
        else:
            if len(non_zero_indices[0]) < NUM_MIN_POINTS_PEDESTRIAN:
                bounding_boxes_without_points.append(i)
    
    eliminate_lines_from_file(BB_path, new_path_output, BB_file, bounding_boxes_without_points)

def eliminate_BB(path_lidar, path_BB, new_path_output, lidar_number):
    
    # Replace 'X' in the paths with the lidar_number
    path_lidar = path_lidar.replace('X', str(lidar_number))
    path_BB = path_BB.replace('X', str(lidar_number))
    new_path_output = new_path_output.replace('X', str(lidar_number))

    # Get the list of files in the specified folder
    lidar_files = sorted([f for f in os.listdir(path_lidar) if f.endswith('.csv')])
    BB_files = sorted([f for f in os.listdir(path_BB) if f.endswith('.csv')])

    # Create the new folder if it doesn't exist
    if not os.path.exists(new_path_output):
        os.makedirs(new_path_output)

    # Use multiprocessing to process files in parallel
    with Pool() as pool:
        pool.map(process_file, [(path_lidar, path_BB, new_path_output, lidar_file, BB_file) for lidar_file, BB_file in zip(lidar_files, BB_files)])

if __name__ == "__main__":

    while True:
        user_input = input("Enter the number of the lidar for the single lidar, or enter 'all' to process all the lidar: ")
        if user_input == 'all':
            for i in range(NUMBER_OF_SENSORS):
                eliminate_BB(LIDAR_X_GRID_DIRECTORY, NEW_POSITIONS_LIDAR_X_GRID_DIRECTORY, POSITION_LIDAR_X_GRID_NO_BB, i+1)
                print("lidar" + str(i+1) + " done")
            break
        elif (int(user_input) in range(1, NUMBER_OF_SENSORS+1)):
            eliminate_BB(LIDAR_X_GRID_DIRECTORY, NEW_POSITIONS_LIDAR_X_GRID_DIRECTORY, POSITION_LIDAR_X_GRID_NO_BB, int(user_input))
            break
        else:
            print("Invalid input.")