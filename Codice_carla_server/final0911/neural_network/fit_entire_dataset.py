import os
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from functions_for_NN import generate_combined_grid_maps, number_of_BB
from constants import *
from functions_for_NN import split_data
import pickle
import random

def fit_scalers():
    # Initialize the scalers
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Parameters for processing
    number_of_chucks = 2

    random.seed(SEED)

    # Shuffle files_lidar_1 and files_BB_1 in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_1_GRID_DIRECTORY)]), sorted([f for f in os.listdir(POSITION_LIDAR_1_GRID_NO_BB)])))
    random.shuffle(combined_files)
    files_lidar_1, files_BB_1 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_1 = list(files_lidar_1)
    files_BB_1 = list(files_BB_1)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, POSITION_LIDAR_1_GRID_NO_BB)
    print(f"\nSum_complete_lidar1: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_2_GRID_DIRECTORY)]), sorted([f for f in os.listdir(POSITION_LIDAR_2_GRID_NO_BB)])))
    random.shuffle(combined_files)
    files_lidar_2, files_BB_2 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_2 = list(files_lidar_2)
    files_BB_2 = list(files_BB_2)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_2, POSITION_LIDAR_2_GRID_NO_BB)
    print(f"\nSum_complete_lidar2: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar2: ", sum_ped/len(files_BB_2), sum_bic/len(files_BB_2), sum_car/len(files_BB_2))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_3_GRID_DIRECTORY)]), sorted([f for f in os.listdir(POSITION_LIDAR_3_GRID_NO_BB)])))
    random.shuffle(combined_files)
    files_lidar_3, files_BB_3 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_3 = list(files_lidar_3)
    files_BB_3 = list(files_BB_3)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, POSITION_LIDAR_3_GRID_NO_BB)
    print(f"\nSum_complete_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)

    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    # Number of files of each chunk for each lidar
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chucks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chucks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chucks) #type: ignore

    print("Number of files for each chunck: ", file_for_chunck1, file_for_chunck2, file_for_chunck3)

    for i in range(number_of_chucks): #type: ignore
        complete_grid_maps = []
        complete_grid_maps_BB = []

        print(f"\nChunck number {i+1} of {number_of_chucks}: ")

        files_lidar_chunck = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
        files_BB_chunck = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, False) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_LIDAR_1_GRID_NO_BB)
        print(f"\nSum_chunck_lidar1: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_lidar1: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        files_lidar_chunck = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
        files_BB_chunck = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, False) # type: ignore
        
        # Info for lidar 2 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_LIDAR_2_GRID_NO_BB)
        print(f"\nSum_chunck_lidar2: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_lidar2: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        files_lidar_chunck = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
        files_BB_chunck = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, False) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_LIDAR_3_GRID_NO_BB)
        print(f"\nSum_chunck_lidar3: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_lidar3: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

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
