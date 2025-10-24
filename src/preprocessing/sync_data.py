"""
Position Synchronization and Processing Module

This module handles the synchronization of LiDAR sensor data with corresponding 
bounding box position data based on timestamps. It finds the closest temporal 
matches between LiDAR captures and position snapshots, then applies coordinate 
transformations to align different sensor positions.
"""

# Standard library imports
import sys
import os
import datetime
import json
import shutil
from multiprocessing import Pool
from itertools import chain

# Third-party imports
import pandas as pd

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants and functions
from config import *

def get_sensor_offsets_from_json(number_lidar):
    """
    Load per-sensor XYZ offsets from SENSORS_DIR/lidar{number_lidar}.json,
    divide each by GRID_RESOLUTION, and invert the sign.

    Returns:
        (ox, oy, oz) as floats
    Raises:
        FileNotFoundError / KeyError / ValueError if the file or fields are missing.
    """
    cfg_path = os.path.join(SENSORS_DIR, f"lidar{number_lidar}.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expected structure:
    # data["actors"][0]["location"] -> {"x": ..., "y": ..., "z": ...}
    try:
        loc = data["actors"][0]["location"]
        x = float(loc["x"])
        y = float(loc["y"])
        z = float(loc["z"])
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid JSON structure in {cfg_path}: {e}")

    # divide by GRID_RESOLUTION and invert sign
    if GRID_RESOLUTION == 0:
        raise ValueError("GRID_RESOLUTION must be non-zero.")

    ox = -x
    oy = -y
    oz = -z
    return ox, oy, oz


def preprocessing_data(path_lidar, new_positions_lidar_output, lidar_number):
    """
    Main preprocessing function that coordinates the synchronization of LiDAR data 
    with position data for a specific sensor.
    
    This function processes a single LiDAR sensor by:
    1. Replacing placeholder 'X' with the actual lidar number in paths
    2. Getting list of LiDAR files
    3. Creating output directory if needed
    4. Finding and copying matching position files
    5. Applying coordinate transformations
    
    Args:
        path_lidar (str): Template path to LiDAR data directory (contains 'X' placeholder)
        new_positions_lidar_output (str): Template path for output directory (contains 'X' placeholder)
        lidar_number (int): Number of the LiDAR sensor (1-based indexing)
    
    Returns:
        None
    """
    # Replace 'X' placeholder in the paths with the actual lidar sensor number
    path_lidar = path_lidar.replace('X', str(lidar_number))
    new_positions_lidar_output = new_positions_lidar_output.replace('X', str(lidar_number))

    # Get list of LiDAR files (without extension) and sort them chronologically
    files_in_lidar_output_removed = sorted([f[:-4] for f in os.listdir(path_lidar) if os.path.isfile(os.path.join(path_lidar, f))])
    
    # Create the output directory if it doesn't already exist
    if not os.path.exists(new_positions_lidar_output):
        os.makedirs(new_positions_lidar_output)
    
    # Find and copy the closest matching position files for each LiDAR file
    new_file_names_lidar_output = compare_and_save_positions(files_in_lidar_output_removed, new_positions_lidar_output)
    
    # Apply coordinate transformations to the copied position files
    modify_positions(new_file_names_lidar_output, new_positions_lidar_output, lidar_number)

# Timestamp comparison utility function
def diff(date1, date2):
    """
    Calculate the time difference between two timestamp strings.
    
    This function parses two datetime strings in a specific format and returns
    the time difference between them as a timedelta object.
    
    Args:
        date1 (str): First timestamp in format "%Y%m%d_%H%M%S_%f"
        date2 (str): Second timestamp in format "%Y%m%d_%H%M%S_%f"
    
    Returns:
        datetime.timedelta: Time difference between date2 and date1
    """
    format = "%Y%m%d_%H%M%S_%f"
    datetime1 = datetime.datetime.strptime(date1, format)
    datetime2 = datetime.datetime.strptime(date2, format)
    diff = datetime2 - datetime1
    return diff

# Main synchronization function
def compare_and_save_positions(lidar_files, new_position_path):
    """
    Find the closest temporal match between LiDAR data and position data, then copy 
    the matching position files to a new directory.
    
    This function implements a temporal synchronization algorithm that:
    1. For each LiDAR file timestamp, finds the position file with the closest timestamp
    2. Uses an optimization to avoid re-scanning already processed position files
    3. Copies the matched position files to a new directory with sequential naming
    4. Applies coordinate transformations based on sensor offset
    
    The algorithm assumes both lists are sorted chronologically and uses the fact that
    if LiDAR file N matches with position file M, then LiDAR file N+1 will likely
    match with position file M or later.
    
    Args:
        lidar_files (list): Sorted list of LiDAR file timestamps (without extensions)
        new_position_path (str): Path where matched position files will be copied
    
    Returns:
        list: List of new filenames for the copied position files
        
    Raises:
        SystemExit: If the number of LiDAR files doesn't match position files found
    """
    before_file = ''  # Stores the best match found for current LiDAR file
    positions_files = []  # List of matched position filenames
    last_position = 0  # Optimization: start searching from last matched position
    
    # Iterate through each LiDAR file to find its closest position match
    for file_lidar in lidar_files:
        # Initialize with a very large time difference (1000 days)
        before_diff = datetime.timedelta(days=1000)
        before_diff = abs(before_diff.total_seconds())

        # Special case: if the previous best match was the last position file,
        # we need to add it to the list (since the loop below won't trigger)
        if before_file == files_in_position_removed[-1]:
            positions_files.append(before_file)
        
        before_file = ''  # Reset for current LiDAR file
        
        # Search through position files starting from last_position (optimization)
        for i in range(last_position, len(files_in_position_removed)):
            # Calculate absolute time difference between LiDAR and position timestamps
            difference = diff(file_lidar, files_in_position_removed[i])
            difference = abs(difference.total_seconds())
            
            # If this position file is closer in time, update the best match
            if difference <= before_diff:
                before_diff = difference
                before_file = files_in_position_removed[i]
            else:
                # If difference starts increasing, we found the best match
                # Add it to the list and update search start position
                positions_files.append(before_file)
                last_position = i - 1  # Start next search from previous position
                break

    # Handle edge case: if the last LiDAR file's best match is the last position file
    if before_file == files_in_position_removed[-1]:
        positions_files.append(before_file)

    # Validation: ensure we found a position match for every LiDAR file
    try:
        if len(lidar_files) == len(positions_files):
            pass  # All good, continue processing
    except ValueError:
        print('THE TWO LISTS HAVE DIFFERENT LENGTHS')
        sys.exit(1)

    # Add .csv extension to all matched position filenames
    complete_file_name = [name + ".csv" for name in positions_files]

    # Find the correct snapshot directory by checking which one contains our files
    z = 1
    while(True):
        position_path = SNAPSHOT_DIRECTORY.replace('X', str(z))       
        source_file = os.path.join(position_path, complete_file_name[0])
        if os.path.exists(source_file):  # Check if file exists before copying
            break
        else:
            z += 1
    
    # Copy matched position files to new directory with sequential naming
    new_file_names = []
    for i, file_name in enumerate(complete_file_name):
        # Construct the full source and destination file paths
        source_file = os.path.join(position_path, file_name)
        # Create new filename with sequential index to avoid conflicts
        new_file_name = f"{file_name[:-4]}_{i}.csv"
        new_file_names.append(new_file_name)
        destination_file = os.path.join(new_position_path, new_file_name)
        
        # Copy the file if it exists
        if os.path.exists(source_file):  
            shutil.copy(source_file, destination_file)
        else:
            print(f"File not found: {file_name}")
            
    return new_file_names

def modify_position_file(args):
    
    file, new_path_position, offsets = args
    csv_path = os.path.join(new_path_position, file)
    df = pd.read_csv(csv_path)

    cols_to_modify = df.columns[2:]
    if len(cols_to_modify) < 3:
        raise ValueError(f"Expected at least three coordinate columns after metadata in {csv_path}.")

    ox, oy, oz = offsets
    
    # Apply coordinate offsets
    # X, Y, Z columns are assumed to be the first three after the two metadata columns
    df[cols_to_modify[0]] = df[cols_to_modify[0]] + ox  # X
    df[cols_to_modify[1]] = df[cols_to_modify[1]] + oy  # Y
    df[cols_to_modify[2]] = df[cols_to_modify[2]] + oz  # Z

    # Save the modified DataFrame back to the same CSV file
    df.to_csv(csv_path, index=False)


def modify_positions(new_file_names, new_path_position, number_lidar):
    offsets = get_sensor_offsets_from_json(number_lidar)
    with Pool() as pool:
        pool.map(
            modify_position_file,
            [(file, new_path_position, offsets) for file in new_file_names]
        )

# Main execution block
if __name__ == "__main__":
    
    # Discover all snapshot directories and collect position file timestamps
    k = 1  # Counter for snapshot directory numbering
    files_in_position_removed = []  # List to store all position file timestamps
    
    # Loop through numbered snapshot directories until no more are found
    while(True):
        # Replace 'X' placeholder with current directory number
        path_position = SNAPSHOT_DIRECTORY.replace('X', str(k))       
        
        # If directory doesn't exist, we've found all snapshot directories
        if not os.path.exists(path_position):
            break 
        else:
            # Get all CSV files in this directory (remove .csv extension for timestamp comparison)
            # Sort them and add to the master list
            files_in_position_removed.append(sorted([f[:-4] for f in os.listdir(path_position) if os.path.isfile(os.path.join(path_position, f))]))
            k += 1
    
    # Flatten the list of lists and sort all position timestamps chronologically
    # This creates a master timeline of all available position data
    files_in_position_removed = sorted(list(chain.from_iterable(files_in_position_removed)))

    # Process each LiDAR sensor sequentially
    for i in range(NUMBER_OF_SENSORS):
        # Process sensor number (i+1) since sensors are 1-indexed
        preprocessing_data(LIDAR_DIRECTORY, SNAPSHOT_SYNCRONIZED_DIRECTORY, i+1)
        # Provide progress feedback to user
        print("lidar" + str(i+1) + " completed")