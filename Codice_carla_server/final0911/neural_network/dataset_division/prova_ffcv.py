from ffcv.writer import DatasetWriter
from ffcv.fields import *
import pickle
import os
import random
import math
import numpy as np
import gc
from concurrent.futures import ThreadPoolExecutor
from constants import *
from functions_for_NN import *

number_of_chuncks = NUMBER_OF_CHUNCKS
number_of_chuncks_test = NUMBER_OF_CHUNCKS_TEST

if __name__ == "__main__":
    '''

    gc.collect()

    set_start_method("spawn", force=True)

    random.seed(SEED)

    os.makedirs(CHUNCKS_DIR, exist_ok=True)

    # Load scalers
    with open(os.path.join(SCALER_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(SCALER_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    # Generation chuncks training set

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

    del combined_files
    gc.collect()

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, POSITION_LIDAR_3_GRID_NO_BB)
    print(f"\nSum_complete_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    # Number of files of each chunck for each lidar
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chuncks) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chuncks) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chuncks) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    for i in range(number_of_chuncks): #type: ignore
        
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_numb_BB = []

        print(f"\nChunck number {i+1} of {number_of_chuncks}")

        files_lidar_chunck_1 = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
        files_BB_chunck_1 = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
        
        files_lidar_chunck_2 = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
        files_BB_chunck_2 = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore

        files_lidar_chunck_3 = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
        files_BB_chunck_3 = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            futures.append(executor.submit(process_lidar_chunk, LIDAR_1_GRID_DIRECTORY, POSITION_LIDAR_1_GRID_NO_BB, files_lidar_chunck_1, files_BB_chunck_1, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))
            futures.append(executor.submit(process_lidar_chunk, LIDAR_2_GRID_DIRECTORY, POSITION_LIDAR_2_GRID_NO_BB, files_lidar_chunck_2, files_BB_chunck_2, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))
            futures.append(executor.submit(process_lidar_chunk, LIDAR_3_GRID_DIRECTORY, POSITION_LIDAR_3_GRID_NO_BB, files_lidar_chunck_3, files_BB_chunck_3, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False))

            for future in futures:
                complete_grid_maps, complete_grid_maps_BB = future.result()

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)
        print(f"\ncomplete grid map shape : {complete_grid_maps.shape}")

        # Concatenate the lists in complete_grid_maps_BB along the first dimension
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        # Number of random samples you want to take
        num_samples = int(complete_grid_maps.shape[0] * 0.2)

        # Generate random indices
        random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

        # Select elements at the random indices
        random_complete_grid_maps = complete_grid_maps[random_indices]
        random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]

        print(f"\nRandom complete grid maps shape: {random_complete_grid_maps.shape}")
        print(f"Random complete grid maps BB shape: {random_complete_grid_maps_BB.shape}")
        
        augmented_grid_maps, augmented_grid_maps_BB = apply_augmentation(random_complete_grid_maps, random_complete_grid_maps_BB)

        augmented_grid_maps = np.array(augmented_grid_maps)
        augmented_grid_maps_BB = np.array(augmented_grid_maps_BB)

        print(f"\nAugmented grid maps shape: {augmented_grid_maps.shape}")
        print(f"Augmented grid maps BB shape: {augmented_grid_maps_BB.shape}")

            # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.concatenate((complete_grid_maps, augmented_grid_maps), axis=0)
        print(f"\nNew complete grid map shape : {complete_grid_maps.shape}")
        complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)
        print(f"New complete grid map BB shape : {complete_grid_maps_BB.shape}")

        del augmented_grid_maps, augmented_grid_maps_BB, random_complete_grid_maps, random_complete_grid_maps_BB
        gc.collect()

        indices = np.arange(complete_grid_maps.shape[0])
        np.random.shuffle(indices)
        complete_grid_maps = complete_grid_maps[indices]
        complete_grid_maps_BB = complete_grid_maps_BB[indices]

        complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

        print("shape after expand_dims: ", complete_grid_maps.shape, complete_grid_maps_BB.shape)
        '''
    # Your dataset (`torch.utils.data.Dataset`) of (image, label) pairs
    chunck_dir = '/mnt/c/Users/marco/Desktop/Tesi/Codice_carla/final0911/chuncks'
    dataset = LidarDataset(chunck_dir, chunck_dir, 0)
    
    print(f"Dataset type: {type(dataset)}")  # Print the type of the dataset
    print("lenght of dataset: ", len(dataset))

    # Retrieve and print the first 10 elements from the dataset
    for i in range(10):
        image, label = dataset[i]
        print(f"Element {i}:")
        print(f"Image shape: {image.shape}, Label shape: {label.shape}")

    path = '/mnt/c/Users/marco/Desktop/Tesi/Codice_carla/final0911/ffcvs'
    name = 'ds.beton'

    complete_path = os.path.join(path, name)
    os.makedirs(path, exist_ok=True)

    # Define the fields for the dataset
    writer = DatasetWriter(complete_path, {
    # Tune options to optimize dataset size, throughput at train-time
    'image': NDArrayField(shape=(400,400,), dtype=np.dtype('float32')),  # Use GrayscaleImageField for 1 channel image
    'label': NDArrayField(shape=(400,400,), dtype=np.dtype('float32'))   # Same for labels
    })


    # Write dataset
    writer.from_indexed_dataset(dataset)