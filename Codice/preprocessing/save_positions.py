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
from multiprocessing import Pool
from itertools import chain

def preprocessing_data(path_lidar, new_positions_lidar_output, lidar_number):
    # Replace 'X' in the paths with the lidar_number
    path_lidar = path_lidar.replace('X', str(lidar_number))
    new_positions_lidar_output = new_positions_lidar_output.replace('X', str(lidar_number))

    files_in_lidar_output_removed = sorted([f[:-4] for f in os.listdir(path_lidar) if os.path.isfile(os.path.join(path_lidar, f))])
    
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_positions_lidar_output):
        os.makedirs(new_positions_lidar_output)
    
    new_file_names_lidar_output = compare_and_save_positions(files_in_lidar_output_removed, new_positions_lidar_output)
    modify_positions(new_file_names_lidar_output, new_positions_lidar_output, lidar_number)

#compare two date and return the difference
def diff(date1, date2):
    format = "%Y%m%d_%H%M%S_%f"
    datetime1 = datetime.datetime.strptime(date1, format)
    datetime2 = datetime.datetime.strptime(date2, format)
    diff = datetime2 - datetime1
    return diff

#Compare each lidar data with the positional data (position of the bounding boxes) and find the closest one respect the time, then save these positional data in a new folder(the lidar folder and the new positional folder will have the same amount of data)
def compare_and_save_positions(lidar_files, new_position_path):
    before_file = ''
    positions_files = []
    last_position = 0
    for file_lidar in lidar_files:
        before_diff = datetime.timedelta(days=1000)
        before_diff = abs(before_diff.total_seconds())

        #Without this 'if', in case the best position file is the last one, it will not be saved in the list
        if before_file == files_in_position_removed[-1]:
            positions_files.append(before_file)
        
        before_file = ''
        for i in range(last_position, len(files_in_position_removed)):
            difference = diff(file_lidar, files_in_position_removed[i])
            difference = abs(difference.total_seconds())
            if difference <= before_diff:
                before_diff = difference
                before_file = files_in_position_removed[i]
            else:
                positions_files.append(before_file)
                last_position = i - 1
                break

    #Without this 'if', in case the best position file of the last lidar file is the last one, it will not be saved in the list
    if before_file == files_in_position_removed[-1]:
        positions_files.append(before_file)

    #ERROR PRINT
    try:
        if len(lidar_files) == len(positions_files):
            pass
    except ValueError:
        print('THE TWO LISTS HAVE DIFFERENT LENGTHS')
        sys.exit(1)

    complete_file_name = [name + ".csv" for name in positions_files]

    z = 1
    while(True):
        
        position_path = POSITIONS_DIRECTORY.replace('X', str(z))       
        source_file = os.path.join(position_path, complete_file_name[0])
        if os.path.exists(source_file):  # Check if file exists before copying
            break
        else:
            z += 1
    
    new_file_names = []
    for i, file_name in enumerate(complete_file_name):
        # Construct the full file paths
        source_file = os.path.join(position_path, file_name)
        new_file_name = f"{file_name[:-4]}_{i}.csv"
        new_file_names.append(new_file_name)
        destination_file = os.path.join(new_position_path, new_file_name)
        
        # Copy the file
        if os.path.exists(source_file):  # Check if file exists before copying
            shutil.copy(source_file, destination_file)
        else:
            print(f"File not found: {file_name}")
    return new_file_names

def modify_position_file(args):
    file, new_path_position, number_lidar = args
    csv_path = os.path.join(new_path_position, file)
    df = pd.read_csv(csv_path)

    # Select all columns except the first two
    cols_to_modify = df.columns[2:]

    # Apply the transformations 
    df[cols_to_modify[0]] = df[cols_to_modify[0]] + NEW_POSITIONS_OFFSETS[number_lidar-1][0]
    df[cols_to_modify[1]] = df[cols_to_modify[1]] + NEW_POSITIONS_OFFSETS[number_lidar-1][1]
    df[cols_to_modify[2]] = df[cols_to_modify[2]] + NEW_POSITIONS_OFFSETS[number_lidar-1][2]

    df.to_csv(csv_path, index=False)

def modify_positions(new_file_names, new_path_position, number_lidar):
    with Pool() as pool:
        pool.map(modify_position_file, [(file, new_path_position, number_lidar) for file in new_file_names])

if __name__ == "__main__":

    k = 1
    files_in_position_removed = []
    while(True):
        path_position = POSITIONS_DIRECTORY.replace('X', str(k))       
        if not os.path.exists(path_position):
            break 
        else:
            files_in_position_removed.append(sorted([f[:-4] for f in os.listdir(path_position) if os.path.isfile(os.path.join(path_position, f))]))
            k += 1
    
    files_in_position_removed = sorted(list(chain.from_iterable(files_in_position_removed)))

    while True:
        user_input = input("Enter the number of the lidar for the single lidar, or enter 'all' to process all the lidar: ")
        if user_input == 'all':
            for i in range(NUMBER_OF_SENSORS):
                preprocessing_data(LIDAR_X_DIRECTORY, NEW_POSITION_LIDAR_X_DIRECTORY, i+1)
                print("lidar" + str(i+1) + " done")
            break
        elif (int(user_input) in range(1, NUMBER_OF_SENSORS+1)):
            preprocessing_data(LIDAR_X_DIRECTORY, NEW_POSITION_LIDAR_X_DIRECTORY, int(user_input))
            break
        else:
            print("Invalid input.")