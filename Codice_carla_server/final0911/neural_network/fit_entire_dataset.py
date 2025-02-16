import os
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from functions_for_NN import generate_combined_grid_maps, number_of_BB, visualize_proportion
from constants import *
from functions_for_NN import split_data
import pickle

def fit_scalers():
    # Initialize the scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Parameters for processing
    number_of_chucks = 5

    # Load all the files for lidar 1
    files_lidar_1 = sorted([f for f in os.listdir(LIDAR_1_GRID_DIRECTORY)]) #type: ignore
    files_BB_1 = sorted([f for f in os.listdir(POSITION_LIDAR_1_GRID_NO_BB)]) #type: ignore
    number_BB_1 = number_of_BB(files_BB_1, POSITION_LIDAR_1_GRID_NO_BB)

    # Load all the files for lidar 2
    files_lidar_2 = sorted([f for f in os.listdir(LIDAR_2_GRID_DIRECTORY)]) #type: ignore
    files_BB_2 = sorted([f for f in os.listdir(POSITION_LIDAR_2_GRID_NO_BB)]) #type: ignore
    number_BB_2 = number_of_BB(files_BB_2, POSITION_LIDAR_2_GRID_NO_BB)

    # Load all the files for lidar 3
    files_lidar_3 = sorted([f for f in os.listdir(LIDAR_3_GRID_DIRECTORY)]) #type: ignore
    files_BB_3 = sorted([f for f in os.listdir(POSITION_LIDAR_3_GRID_NO_BB)]) #type: ignore
    number_BB_3 = number_of_BB(files_BB_3, POSITION_LIDAR_3_GRID_NO_BB)

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)

    print(f"Total number of files for lidar 1: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    # Number of files of each chunk for each lidar
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chucks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chucks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chucks) #type: ignore

    for i in range(number_of_chucks): #type: ignore
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_num_BB = []

        print("\nChunck number: ", i+1)

        # Because the last chunck may have less files
        files_for_this_chunck1 = min(file_for_chunck1, len(files_lidar_1))
        print(f"\nNumber of files for chunck number {i+1} of lidar 1: {files_for_this_chunck1}")

        # The if is required because if number_of_chucks is 1, the function split_data will not work
        if ((number_of_chucks > 1) and (i != number_of_chucks - 1)):
            files_lidar_1, files_lidar_chunck, files_BB_1, files_BB_chunck, number_BB_1, number_BB_chunck = split_data(files_lidar_1, files_BB_1, number_BB_1, files_for_this_chunck1)
        else:
            files_lidar_chunck = files_lidar_1
            files_BB_chunck = files_BB_1
            number_BB_chunck = number_BB_1
        generate_combined_grid_maps(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = visualize_proportion(number_BB_chunck)
        print(f"\nSum_chunck_lidar1: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_lidar1: ", sum_ped/len(number_BB_chunck), sum_bic/len(number_BB_chunck), sum_car/len(number_BB_chunck))

        # Because the last chunck may have less files
        files_for_this_chunck2 = min(file_for_chunck2, len(files_lidar_2))
        print(f"\nNumber of files for chunck number {i+1} of lidar 2: {files_for_this_chunck2}")

        if ((number_of_chucks > 1) and (i != number_of_chucks - 1)):
            files_lidar_2, files_lidar_chunck, files_BB_2, files_BB_chunck, number_BB_2, number_BB_chunck = split_data(files_lidar_2, files_BB_2, number_BB_2, files_for_this_chunck2)
        else:
            files_lidar_chunck = files_lidar_2
            files_BB_chunck = files_BB_2
            number_BB_chunck = number_BB_2
        generate_combined_grid_maps(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore
        
        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = visualize_proportion(number_BB_chunck)
        print(f"\nSum_chunck_lidar2: ", sum_ped, sum_bic, sum_car)
        print(f"Averag_chunck_lidar2: ", sum_ped/len(number_BB_chunck), sum_bic/len(number_BB_chunck), sum_car/len(number_BB_chunck))

        # Because the last chunck may have less files
        files_for_this_chunck3 = min(file_for_chunck3, len(files_lidar_3))
        print(f"\nNumber of files for chunck number {i+1} of lidar 2: {files_for_this_chunck3}")

        if ((number_of_chucks > 1) and (i != number_of_chucks - 1)):
            files_lidar_3, files_lidar_chunck, files_BB_3, files_BB_chunck, number_BB_3, number_BB_chunck = split_data(files_lidar_3, files_BB_3, number_BB_3, files_for_this_chunck3)
        else:
            files_lidar_chunck = files_lidar_3
            files_BB_chunck = files_BB_3
            number_BB_chunck = number_BB_3
        generate_combined_grid_maps(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_num_BB) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = visualize_proportion(number_BB_chunck)
        print(f"Sum_chunck_lidar3: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_lidar3: ", sum_ped/len(number_BB_chunck), sum_bic/len(number_BB_chunck), sum_car/len(number_BB_chunck))

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
