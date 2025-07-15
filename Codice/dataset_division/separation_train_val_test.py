import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
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
        #print(f"Moved file: {file}")
        shutil.move(BB_directory_initial, BB_directory_finish)
        #print(f"Moved file: {file_BB}")
    except FileNotFoundError:
        print(f"File not found: {file}")

def move_file(files_lidar, files_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path):
    with Pool() as pool:
        pool.starmap(cut_files, [(file, file_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path) for file, file_BB in zip(files_lidar, files_BB)])
    
def main_loop(lidar_path_initial, BB_path_initial, lidar_path_final_test, BB_path_final_test, lidar_path_final_val, BB_path_final_val):
    
    # Shuffle files_lidar_1 and files_BB_1 in the same way
    files_lidar = sorted([f for f in os.listdir(lidar_path_initial)]) 
    files_BB = sorted([f for f in os.listdir(BB_path_initial)])
    
    sum_ped, sum_bic, sum_car = number_of_BB(files_BB, BB_path_initial)
    print(f"\nSum_complete_lidar: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar: ", sum_ped/len(files_BB), sum_bic/len(files_BB), sum_car/len(files_BB))

    # Total number of files for each lidar
    total_num_of_files = len(files_lidar)

    file_for_test = math.ceil(total_num_of_files * TEST_SIZE) #type: ignore

    files_lidar_test = files_lidar[ 0 : min( file_for_test, len(files_lidar) ) ] #type: ignore
    files_BB_test = files_BB[ 0 : min( file_for_test, len(files_BB) ) ] #type: ignore

    files_lidar_val = files_lidar[ min( file_for_test, len(files_lidar) ) : 2*min( file_for_test, len(files_lidar) ) ] #type: ignore
    files_BB_val = files_BB[ min( file_for_test, len(files_BB) ) : 2*min( file_for_test, len(files_lidar) ) ] #type: ignore

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_test, BB_path_initial)
    print(f"\nSum_lidar_test: ", sum_ped, sum_bic, sum_car)
    print(f"Average_lidar_test: ", sum_ped/len(files_BB_test), sum_bic/len(files_BB_test), sum_car/len(files_BB_test))

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_val, BB_path_initial)
    print(f"\nSum_lidar_val: ", sum_ped, sum_bic, sum_car)
    print(f"Average_lidar_val: ", sum_ped/len(files_BB_val), sum_bic/len(files_BB_val), sum_car/len(files_BB_val))

    
    move_file(files_lidar_test, files_BB_test, lidar_path_initial, BB_path_initial, lidar_path_final_test, BB_path_final_test) #type: ignore
    move_file(files_lidar_val, files_BB_val, lidar_path_initial, BB_path_initial, lidar_path_final_val, BB_path_final_val) #type: ignore

if __name__=="__main__":

    for i in range(1, NUMBER_OF_SENSORS+1):

        print(f"Lidar{i}")
        lidar_path = LIDAR_X_GRID_DIRECTORY.replace("X", str(i))
        position_path = SNAPSHOT_X_GRID_DIRECTORY.replace("X", str(i))
        lidar_test_path = LIDAR_X_TEST.replace("X", str(i))
        position_test_path = POSITION_X_TEST.replace("X", str(i))
        lidar_val_path = LIDAR_X_VAL.replace("X", str(i))
        position_val_path = POSITION_X_VAL.replace("X", str(i))
        main_loop(lidar_path, position_path, lidar_test_path, position_test_path, lidar_val_path, position_val_path)
        print("\n")

    










