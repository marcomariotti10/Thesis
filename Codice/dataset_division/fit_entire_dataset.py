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
from concurrent.futures import ThreadPoolExecutor

def process_lidar_chunk(lidar_directory, files_lidar_chunck, complete_grid_maps):
    generate_combined_grid_maps_fit(lidar_directory, files_lidar_chunck, complete_grid_maps) # type: ignore

    return complete_grid_maps

def fit_scalers(lidar_paths):
    # Initialize the scalers
    scaler_X = MinMaxScaler()

    # Parameters for processing
    number_of_chucks = NUMBER_OF_CHUNCKS

    random.seed(SEED)

    files_lidar = []
    files_for_chunck = []

    for i in range(NUMBER_OF_SENSORS):

        # Shuffle files_lidar and files_BB in the same way
        files_lidar_1 = list((sorted([f for f in os.listdir(lidar_paths[i])])))
        random.shuffle(files_lidar_1)
        # Convert back to lists if needed
        files_lidar.append(list(files_lidar_1))

    for i in range(NUMBER_OF_SENSORS):
        files_for_chunck.append(math.ceil(len(files_lidar[i]) / number_of_chucks)) #type: ignore
        print(f"Number of files of lidar {i+1} and for each chunk: {len(files_lidar[i])}    {files_for_chunck[i]}")

    for i in range(number_of_chucks): #type: ignore

        print(f"\nFitting chunck number {i+1} ")
        
        complete_grid_maps = []

        all_files_chunck = []

        for k in range(NUMBER_OF_SENSORS):

            files_lidar_chunck = files_lidar[k][ i*files_for_chunck[k] : min( (i+1)*files_for_chunck[k], len(files_lidar[k]) ) ] #type: ignore
            all_files_chunck.append(files_lidar_chunck)

        with ThreadPoolExecutor(max_workers=NUMBER_OF_SENSORS) as executor:
            futures = []
            for j in range(NUMBER_OF_SENSORS):
                futures.append(executor.submit(process_lidar_chunk, lidar_paths[j], all_files_chunck[j], complete_grid_maps))
            
            for future in futures:
                complete_grid_maps = future.result()
        
        # Shuffle the data
        random.shuffle(complete_grid_maps)

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)

        # Normalize the training data and save the scalers in variables
        scaler_X.partial_fit(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1]))

    # Specify the directory where you want to save the scalers
    save_directory = SCALER_DIR

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)
    print("\n")

    # Save the scalers to the specified directory
    with open(os.path.join(save_directory, 'scaler_X.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    print("scaler_X saved")


if __name__ == "__main__":

    lidar_direcory_list = []

    for i in range (1, NUMBER_OF_SENSORS+1):
        lidar_direcory_list.append(LIDAR_X_GRID_DIRECTORY.replace('X', str(i)))

    fit_scalers(lidar_direcory_list)
