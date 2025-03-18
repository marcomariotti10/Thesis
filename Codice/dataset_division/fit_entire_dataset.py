import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import os
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
from pympler import asizeof


def fit_scalers():
    # Initialize the scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Parameters for processing
    number_of_chucks = NUMBER_OF_CHUNCKS + int(NUMBER_OF_CHUNCKS*0.2)

    random.seed(SEED)

    files_lidar = []
    files_BB = []
    files_for_chunck = []

    for i in range(1, NUMBER_OF_SENSORS+1):
        lidar_path = LIDAR_X_GRID_DIRECTORY.replace('X', str(i))
        position_path = POSITION_LIDAR_X_GRID_NO_BB.replace('X', str(i))

        # Shuffle files_lidar and files_BB in the same way
        combined_files = list(zip(sorted([f for f in os.listdir(lidar_path)]), sorted([f for f in os.listdir(position_path)])))
        random.shuffle(combined_files)
        files_lidar_1, files_BB_1 = zip(*combined_files)
        # Convert back to lists if needed
        files_lidar.append(list(files_lidar_1))
        files_BB.append(list(files_BB_1))

        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, position_path)
        print(f"\nSum_complete_lidar1: ", sum_ped, sum_bic, sum_car)
        print(f"Average_complete_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))


    i = 0

    for i in range(NUMBER_OF_SENSORS):
        print(f"Number of files of lidar {i+1}: {len(files_lidar[i])}")

    i = 0

    for i in range(NUMBER_OF_SENSORS):
        files_for_chunck.append(math.ceil(len(files_lidar[i]) / number_of_chucks)) #type: ignore
        print(f"Number of files of lidar {i+1} for  each chunck: {files_for_chunck[i]}")
    
    i = 0

    for i in range(number_of_chucks): #type: ignore
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_num_BB = []

        print(f"\nChunck number {i+1} of {number_of_chucks}: ")

        for k in range(1, NUMBER_OF_SENSORS+1):
            lidar_path = LIDAR_X_GRID_DIRECTORY.replace('X', str(k))
            position_path = POSITION_LIDAR_X_GRID_NO_BB.replace('X', str(k))

            files_lidar_chunck = files_lidar_1[ i*files_for_chunck[k] : min( (i+1)*files_for_chunck[k], len(files_lidar[k]) ) ] #type: ignore
            files_BB_chunck = files_BB_1[ i*files_for_chunck[k] : min( (i+1)*files_for_chunck[k], len(files_BB[k]) ) ] #type: ignore
            generate_combined_grid_maps(lidar_path, position_path, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_num_BB, False) # type: ignore

            # Info for lidar 1 about the number of bounding boxes
            sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, position_path)
            print(f"\nSum_chunck_lidar1: ", sum_ped, sum_bic, sum_car)
            print(f"Average_chunck_lidar1: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        
        # Shuffle the data
        combined_files = list(zip(complete_grid_maps, complete_grid_maps_BB))
        random.shuffle(combined_files)
        complete_grid_maps, complete_grid_maps_BB = zip(*combined_files)
        # Convert back to lists if needed
        complete_grid_maps = list(complete_grid_maps)
        complete_grid_maps_BB = list(complete_grid_maps_BB)

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)

        # Normalize the training data and save the scalers in variables
        scaler_X.partial_fit(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1]))
        scaler_y.partial_fit(complete_grid_maps_BB.reshape(-1, complete_grid_maps_BB.shape[-1]))

    # Specify the directory where you want to save the scalers
    save_directory = SCALER_DIR

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save the scalers to the specified directory
    with open(os.path.join(save_directory, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    print("scaler_X saved")
    with open(os.path.join(save_directory, 'scaler_y.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    print("scaler_y saved")

if __name__ == "__main__":
    fit_scalers()
