import datetime
import sys
import os
import shutil
import csv
import numpy as np
import pandas as pd
from constants import *

def preprocessing_data(lidar_number):
    
    if lidar_number == 1:
        path_lidar_1 = LIDAR_1_DIRECTORY
        files_in_lidar_output_1 = [f for f in os.listdir(path_lidar_1) if os.path.isfile(os.path.join(path_lidar_1, f))]
        files_in_lidar_output_1_removed = []
        for file in files_in_lidar_output_1:
            file = file[:-11]
            files_in_lidar_output_1_removed.append(file)
        new_positions_lidar_output_1 = NEW_POSITION_LIDAR_1_DIRECTORY
        new_file_names_lidar_output_1 = compare_and_save_positions(files_in_lidar_output_1_removed, new_positions_lidar_output_1)
        modify_positions(new_file_names_lidar_output_1, new_positions_lidar_output_1, 1)
    
    elif lidar_number == 2:
        
        path_lidar_2 = LIDAR_2_DIRECTORY
        files_in_lidar_output_2 = [f for f in os.listdir(path_lidar_2) if os.path.isfile(os.path.join(path_lidar_2, f))]
        files_in_lidar_output_2_removed = []
        for file in files_in_lidar_output_2:
            file = file[:-11]
            files_in_lidar_output_2_removed.append(file)
        new_positions_lidar_output_2 = NEW_POSITION_LIDAR_2_DIRECTORY
        new_file_names_lidar_output_2 = compare_and_save_positions(files_in_lidar_output_2_removed, new_positions_lidar_output_2)
        modify_positions(new_file_names_lidar_output_2, new_positions_lidar_output_2, 2)

    elif lidar_number == 3:

        path_lidar_3 = LIDAR_3_DIRECTORY
        files_in_lidar_output_3 = [f for f in os.listdir(path_lidar_3) if os.path.isfile(os.path.join(path_lidar_3, f))]
        files_in_lidar_output_3_removed = []
        for file in files_in_lidar_output_3:
            file = file[:-11]
            files_in_lidar_output_3_removed.append(file)
        new_positions_lidar_output_3 = NEW_POSITION_LIDAR_3_DIRECTORY
        new_file_names_lidar_output_3 = compare_and_save_positions(files_in_lidar_output_3_removed, new_positions_lidar_output_3)
        modify_positions(new_file_names_lidar_output_3, new_positions_lidar_output_3, 3)
    
    else:
        print("Invalid lidar number.")
        sys.exit(1)


#compare two date and return the difference
def diff(date1, date2):
    format = "%Y%m%d_%H%M%S_%f"
    datetime1 = datetime.datetime.strptime(date1, format)
    datetime2 = datetime.datetime.strptime(date2, format)
    diff = datetime2-datetime1
    return diff

#Compare each lidar data with the positional data (position of the bounding boxes) and find the closest one respect the time, then save these positional data in a new folder(the lidar folder and the new positional folder will have the same amount of data)
def compare_and_save_positions(lidar_files, new_position_path):
    before_file = ''
    positions_files = []
    last_position = 0
    for file_lidar in lidar_files :
        before_diff = datetime.timedelta(days = 1000)
        before_diff = (abs(before_diff.total_seconds()))

        #Without this 'if', in case the best position file is the last one, it will not be saved in the list
        if (before_file == files_in_position_removed[-1]):
            positions_files.append(before_file)
        
        before_file = ''
        for i in range(last_position, len(files_in_position_removed)):
            difference = diff(file_lidar, files_in_position_removed[i])
            difference = (abs(difference.total_seconds()))
            if difference <= before_diff :
                before_diff = difference
                before_file = files_in_position_removed[i]
            else :
                positions_files.append(before_file)
                last_position = i-1
                break

    #Without this 'if', in case the best position file of the last lidar file is the last one, it will not be saved in the list
    if (before_file == files_in_position_removed[-1]):
            positions_files.append(before_file)

    #ERROR PRINT
    try:
        if (len(lidar_files)) == len(positions_files): print("The two lists has the same lenght")
    
    except ValueError:
        print('THE TWO LIST HAS DIFFERENT LENGHTS') 
        sys.exit(1)

    print(positions_files)
    complete_file_name = [name + ".csv" for name in positions_files]

    # Create the new folder if it doesn't exist
    if not os.path.exists(new_position_path):
        os.makedirs(new_position_path)

    new_file_names = []
    for i, file_name in enumerate(complete_file_name):
        # Construct the full file paths
        source_file = os.path.join(path_position, file_name)
        new_file_name = f"{file_name[:-4]}_{i}.csv"
        new_file_names.append(new_file_name)
        destination_file = os.path.join(new_position_path, new_file_name)
        
        # Copy the file
        if os.path.exists(source_file):  # Check if file exists before copying
            shutil.copy(source_file, destination_file)
            print(f"Copied and renamed: {file_name} to {new_file_name}")
        else:
            print(f"File not found: {file_name}")
    return new_file_names

#Convert the world positions in local positions
def modify_positions(new_file_names, new_path_position, number_lidar):
    for file in new_file_names:
        csv_path = os.path.join(new_path_position, file)
        print(f"Loading {file}...")
        df = pd.read_csv(csv_path)

        # Select all columns except the first two
        cols_to_modify = df.columns[2:]

        # Apply the transformations (number_lidar = 1 for lidar1, 2 for lidar2, 3 for lidar3)
        if (number_lidar == 1):
            df[cols_to_modify[0]] = df[cols_to_modify[0]] + 70.15
            df[cols_to_modify[1]] = df[cols_to_modify[1]] + 11.50
            df[cols_to_modify[2]] = df[cols_to_modify[2]] - 6.0
        elif (number_lidar == 2):
            df[cols_to_modify[0]] = df[cols_to_modify[0]] + 100.59
            df[cols_to_modify[1]] = df[cols_to_modify[1]] - 27.46
            df[cols_to_modify[2]] = df[cols_to_modify[2]] - 6.0
        elif (number_lidar == 3):
            df[cols_to_modify[0]] = df[cols_to_modify[0]] + 96.83
            df[cols_to_modify[1]] = df[cols_to_modify[1]] + 6.31
            df[cols_to_modify[2]] = df[cols_to_modify[2]] - 6.0

        df.to_csv(csv_path, index=False)

if __name__ == "__main__":

    path_position = POSITIONS_DIRECTORY
    files_in_position = [f for f in os.listdir(path_position) if os.path.isfile(os.path.join(path_position, f))]
    files_in_position_removed = []
    for file in files_in_position:
        file = file[:-4]
        files_in_position_removed.append(file)

    while True:
        user_input = input("Enter 1 for lidar1, 2 for lidar2, 3 for lidar3, 4 for all: ")
        if user_input == '1':
            preprocessing_data(1)
            break

        elif user_input == '2':
            preprocessing_data(2)
            break

        elif user_input == '3':
            preprocessing_data(3)
            break

        elif user_input == '4':
            preprocessing_data(1)
            preprocessing_data(2)
            preprocessing_data(3)
            break
        else:
            print("Invalid input. Please enter 1, 2, 3 or 4.")