import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from neural_network import *
import csv
import shutil
import numpy as np
import random
import math
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, set_start_method

def cut_files(file, file_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path):
    """
    Move a file from src_path to dest_path.

    :param src_path: Source file path
    :param dest_path: Destination file path
    """
    lidar_directory_initial = os.path.join(src_lidar_path, file)
    BB_directory_initial = os.path.join(src_BB_path, file_BB)

    lidar_directory_finish = os.path.join(dest_lidar_path, file)
    BB_directory_finish = os.path.join(dest_BB_path, file_BB)

    # Create destination directory if it does not exist
    os.makedirs(os.path.dirname(lidar_directory_finish), exist_ok=True)
    os.makedirs(os.path.dirname(BB_directory_finish), exist_ok=True)

    try:
        # Move the file
        shutil.move(lidar_directory_initial, lidar_directory_finish)
        print(f"Moved file: {file}")
        shutil.move(BB_directory_initial, BB_directory_finish)
        print(f"Moved file: {file_BB}")
    except FileNotFoundError:
        print(f"File not found: {file}")

def move_file(files_lidar, files_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path):
    with Pool() as pool:
        pool.starmap(cut_files, [(file, file_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path) for file, file_BB in zip(files_lidar, files_BB)])
    
def main_loop(lidar_path_initial, BB_path_initial, lidar_path_final, BB_path_final):
    
    # Shuffle files_lidar_1 and files_BB_1 in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(lidar_path_initial)]), sorted([f for f in os.listdir(BB_path_initial)])))
    random.shuffle(combined_files)
    files_lidar, files_BB = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar = list(files_lidar)
    files_BB = list(files_BB)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB, BB_path_initial)
    print(f"\nSum_complete_lidar: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar: ", sum_ped/len(files_BB), sum_bic/len(files_BB), sum_car/len(files_BB))

    # Total number of files for each lidar
    total_num_of_files = len(files_lidar)

    file_for_test = math.ceil(total_num_of_files * TEST_SIZE) #type: ignore

    files_lidar_test = files_lidar[ 0 : min( file_for_test, len(files_lidar) ) ] #type: ignore
    files_BB_test = files_BB[ 0 : min( file_for_test, len(files_BB) ) ] #type: ignore

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_test, BB_path_initial)
    print(f"\nSum_lidar_test: ", sum_ped, sum_bic, sum_car)
    print(f"Average_lidar_test: ", sum_ped/len(files_BB_test), sum_bic/len(files_BB_test), sum_car/len(files_BB_test))

    move_file(files_lidar_test, files_BB_test, lidar_path_initial, BB_path_initial, lidar_path_final, BB_path_final) #type: ignore

if __name__=="__main__":

    print("Lidar1")
    main_loop(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, LIDAR_1_TEST, POSITION_1_TEST)
    print("Lidar2")
    main_loop(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, LIDAR_2_TEST, POSITION_2_TEST)
    print("Lidar3")
    main_loop(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, LIDAR_3_TEST, POSITION_3_TEST)

    










