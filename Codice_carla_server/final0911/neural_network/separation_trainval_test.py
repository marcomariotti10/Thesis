from constants import *
import os
import csv
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, set_start_method
from functions_for_NN import number_of_BB, split_data, visualize_proportion

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
        print(f"Moved file from {file}")
        shutil.move(BB_directory_initial, BB_directory_finish)
        print(f"Moved file from {file_BB}")
    except FileNotFoundError:
        print(f"File not found: {file}")

def move_file(files_lidar, files_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path):
    with Pool() as pool:
        pool.starmap(cut_files, [(file, file_BB, src_lidar_path, src_BB_path, dest_lidar_path, dest_BB_path) for file, file_BB in zip(files_lidar, files_BB)])
    
def main_loop(lidar_path_initial, BB_path_initial, lidar_path_final, BB_path_final):
    
    files_lidar = sorted([f for f in os.listdir(lidar_path_initial)]) #type: ignore
    files_BB = sorted([f for f in os.listdir(BB_path_initial)]) #type: ignore
    numbers_of_BB = number_of_BB(files_BB, BB_path_initial)

    files_lidar_trainval, files_lidar_test, files_BB_trainval, files_BB_test, BB_trainval, BB_test = split_data(files_lidar, files_BB, numbers_of_BB, TEST_SIZE)

    sum_ped, sum_bic, sum_car = visualize_proportion(BB_trainval)
    print(f"Sum_train: ", sum_ped, sum_bic, sum_car)
    print(f"Average_sum_train: ", sum_ped/len(BB_trainval), sum_bic/len(BB_trainval), sum_car/len(BB_trainval))

    sum_ped, sum_bic, sum_car = visualize_proportion(BB_test)
    print(f"Sum_test: ", sum_ped, sum_bic, sum_car)
    print(f"Average_sum_test: ", sum_ped/len(BB_test), sum_bic/len(BB_test), sum_car/len(BB_test))

    move_file(files_lidar_test, files_BB_test, lidar_path_initial, BB_path_initial, lidar_path_final, BB_path_final) #type: ignore

if __name__=="__main__":

    main_loop(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, LIDAR_1_TEST, POSITION_1_TEST)
    main_loop(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, LIDAR_2_TEST, POSITION_2_TEST)
    main_loop(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, LIDAR_3_TEST, POSITION_3_TEST)

    










