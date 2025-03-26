import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import datetime
import sys
import os
import shutil
import csv
import numpy as np
import pandas as pd
import threading

def preprocessing_data(path_lidar, new_positions_lidar_output, lidar_number):
    """Process LiDAR data for a given sensor using threading"""
    path_lidar = path_lidar.replace('X', str(lidar_number))
    new_positions_lidar_output = new_positions_lidar_output.replace('X', str(lidar_number))

    files_in_lidar_output = sorted([f[:-4] for f in os.listdir(path_lidar) if os.path.isfile(os.path.join(path_lidar, f))])

    new_file_names_lidar_output = compare_and_save_positions(files_in_lidar_output, new_positions_lidar_output)
    modify_positions(new_file_names_lidar_output, new_positions_lidar_output, lidar_number)

def diff(date1, date2):
    """Calculate time difference between two timestamps"""
    format = "%Y%m%d_%H%M%S_%f"
    datetime1 = datetime.datetime.strptime(date1, format)
    datetime2 = datetime.datetime.strptime(date2, format)
    return abs((datetime2 - datetime1).total_seconds())

def compare_and_save_positions(lidar_files, new_position_path):
    """Find the closest position file for each LiDAR file and save it"""
    positions_files = []
    last_position = 0
    before_file = ''

    for file_lidar in lidar_files:
        before_diff = float('inf')

        if before_file == files_in_position_removed[-1]:
            positions_files.append(before_file)

        for i in range(last_position, len(files_in_position_removed)):
            difference = diff(file_lidar, files_in_position_removed[i])
            if difference <= before_diff:
                before_diff = difference
                before_file = files_in_position_removed[i]
            else:
                positions_files.append(before_file)
                last_position = i - 1
                break

    if before_file == files_in_position_removed[-1]:
        positions_files.append(before_file)

    if len(lidar_files) != len(positions_files):
        print('ERROR: THE TWO LISTS HAVE DIFFERENT LENGTHS')
        return []

    complete_file_names = [name + ".csv" for name in positions_files]

    if not os.path.exists(new_position_path):
        os.makedirs(new_position_path)

    new_file_names = []
    for i, file_name in enumerate(complete_file_names):
        source_file = os.path.join(path_position, file_name)
        new_file_name = f"{file_name[:-4]}_{i}.csv"
        new_file_names.append(new_file_name)
        destination_file = os.path.join(new_position_path, new_file_name)

        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
        else:
            print(f"File not found: {file_name}")

    return new_file_names

def modify_position_file(file, new_path_position, number_lidar):
    """Modify position data in a single file"""
    csv_path = os.path.join(new_path_position, file)
    df = pd.read_csv(csv_path)

    cols_to_modify = df.columns[2:]
    
    df[cols_to_modify[0]] += NEW_POSITIONS_OFFSETS[number_lidar-1][0]
    df[cols_to_modify[1]] += NEW_POSITIONS_OFFSETS[number_lidar-1][1]
    df[cols_to_modify[2]] += NEW_POSITIONS_OFFSETS[number_lidar-1][2]

    df.to_csv(csv_path, index=False)

def modify_positions(new_file_names, new_path_position, number_lidar):
    """Modify multiple position files using threading"""
    threads = []
    for file in new_file_names:
        thread = threading.Thread(target=modify_position_file, args=(file, new_path_position, number_lidar))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    path_position = POSITIONS_DIRECTORY
    files_in_position_removed = sorted([f[:-4] for f in os.listdir(path_position) if os.path.isfile(os.path.join(path_position, f))])

    user_input = input("Enter the number of the LiDAR sensor or 'all' to process all: ")

    if user_input.lower() == 'all':
        sensor_threads = []
        for i in range(1, NUMBER_OF_SENSORS + 1):
            thread = threading.Thread(target=preprocessing_data, args=(LIDAR_X_DIRECTORY, NEW_POSITION_LIDAR_X_DIRECTORY, i))
            sensor_threads.append(thread)
            thread.start()

        for thread in sensor_threads:
            thread.join()

        print("Processing completed for all sensors.")
    
    elif user_input.isdigit() and 1 <= int(user_input) <= NUMBER_OF_SENSORS:
        preprocessing_data(LIDAR_X_DIRECTORY, NEW_POSITION_LIDAR_X_DIRECTORY, int(user_input))

    else:
        print("Invalid input.")