import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import shutil
from multiprocessing import Pool

def cut_files(file, file_BB, initial_lidar_path, initial_BB_path, final_lidar_path, final_BB_path):
    """
    Move a file from src_path to dest_path.

    :param src_path: Source file path
    :param dest_path: Destination file path
    """
    lidar_directory_initial = os.path.join(initial_lidar_path, file)
    BB_directory_initial = os.path.join(initial_BB_path, file_BB)

    lidar_directory_finish = os.path.join(final_lidar_path, file)
    BB_directory_finish = os.path.join(final_BB_path, file_BB)

    try:
        # Move the file
        shutil.move(lidar_directory_finish, lidar_directory_initial)
        print(f"Moved file: {file}")
        shutil.move(BB_directory_finish, BB_directory_initial)
        print(f"Moved file: {file_BB}")
    except FileNotFoundError:
        print(f"File not found: {file}")

def move_file(files_lidar, files_BB, initial_lidar_path, initial_BB_path, final_lidar_path, final_BB_path):
    with Pool() as pool:
        pool.starmap(cut_files, [(file, file_BB, initial_lidar_path, initial_BB_path, final_lidar_path, final_BB_path) for file, file_BB in zip(files_lidar, files_BB)])

def main_loop(lidar_path_initial, BB_path_initial, lidar_path_final, BB_path_final):
    
    files_lidar = sorted([f for f in os.listdir(lidar_path_final)])
    files_BB = sorted([f for f in os.listdir(BB_path_final)])

    move_file(files_lidar, files_BB, lidar_path_initial, BB_path_initial, lidar_path_final, BB_path_final)

def revert(all_lidar_path, all_BB_path):
    for i in range(1, NUMBER_OF_SENSORS+1):
        print(f"Reverting lidar {i}")
        lidar_grid_path  = LIDAR_X_GRID_DIRECTORY.replace("X", str(i))
        BB_grid_path = POSITION_LIDAR_X_GRID_NO_BB.replace("X", str(i))
        main_loop(lidar_grid_path, BB_grid_path, all_lidar_path[i-1], all_BB_path[i-1])    

if __name__=="__main__":

    all_lidar_test_path = []
    all_BB_test_path = []

    all_lidar_val_path = []
    all_BB_val_path = []

    for i in range(1, NUMBER_OF_SENSORS+1):

        all_lidar_test_path.append(LIDAR_X_TEST.replace("X", str(i)))
        all_BB_test_path.append(POSITION_X_TEST.replace("X", str(i)))

        all_lidar_val_path.append(LIDAR_X_VAL.replace("X", str(i)))
        all_BB_val_path.append(POSITION_X_VAL.replace("X", str(i)))

    revert(all_lidar_test_path, all_BB_test_path) #revert test
    revert(all_lidar_val_path, all_BB_val_path) #revert val