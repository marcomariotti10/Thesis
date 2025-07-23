import sys
import os
import threading
import csv
import ast
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from visualization import *

def eliminate_lines_from_file(BB_path, new_path_output, file_name, lines_to_eliminate):
    with open(BB_path, 'r') as file_BB:
        lines = file_BB.readlines()

    header = lines[0]
    remaining_lines = [line for idx, line in enumerate(lines[1:]) if idx not in lines_to_eliminate]

    path = os.path.join(new_path_output, file_name)
    
    with open(path, 'w') as new_file:
        new_file.write(header)
        new_file.writelines(remaining_lines)

def load_bounding_box(csv_file):
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

def process_file(path_lidar, path_BB, new_path_output, lidar_file, BB_file):
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
        grid_map_BB = np.full((Y_RANGE, X_RANGE), 0, dtype=float)
        
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
    
    eliminate_lines_from_file(BB_path, new_path_output, BB_file, bounding_boxes_without_points)

def eliminate_BB(path_lidar, path_BB, new_path_output, lidar_number):
    path_lidar = path_lidar.replace('X', str(lidar_number))
    path_BB = path_BB.replace('X', str(lidar_number))
    new_path_output = new_path_output.replace('X', str(lidar_number))

    lidar_files = sorted([f for f in os.listdir(path_lidar) if f.endswith('.csv')])
    BB_files = sorted([f for f in os.listdir(path_BB) if f.endswith('.csv')])

    os.makedirs(new_path_output, exist_ok=True)

    threads = []
    for lidar_file, BB_file in zip(lidar_files, BB_files):
        thread = threading.Thread(target=process_file, args=(path_lidar, path_BB, new_path_output, lidar_file, BB_file))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    while True:
        user_input = input("Enter the number of the lidar for single processing, or 'all' for all: ")
        if user_input.lower() == 'all':
            sensor_threads = []
            for i in range(1, NUMBER_OF_SENSORS + 1):
                thread = threading.Thread(target=eliminate_BB, args=(LIDAR_X_GRID_DIRECTORY, SNAPSHOT_X_GRID_DIRECTORY, SNAPSHOT_X_GRID_DIRECTORY_NO_BB, i))
                sensor_threads.append(thread)
                thread.start()
            
            for thread in sensor_threads:
                thread.join()
            print("Processing completed for all sensors.")
            break
        elif user_input.isdigit() and 1 <= int(user_input) <= NUMBER_OF_SENSORS:
            eliminate_BB(LIDAR_X_GRID_DIRECTORY, SNAPSHOT_X_GRID_DIRECTORY, SNAPSHOT_X_GRID_DIRECTORY_NO_BB, int(user_input))
            break
        else:
            print("Invalid input.")
