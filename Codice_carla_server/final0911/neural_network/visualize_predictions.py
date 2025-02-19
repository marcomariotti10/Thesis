import torch
import numpy as np
import os
import math
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from functions_for_NN import *
from constants import *

def visualize_prediction(prediction, ground_truth):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Reshape the prediction and ground truth if necessary
    if prediction.ndim == 1:
        prediction = prediction.reshape((20, 20))  # Adjust the shape as needed
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape((20, 20))  # Adjust the shape as needed

    ax[0].imshow(prediction, cmap='gray')
    ax[0].set_title('Prediction')
    ax[1].imshow(ground_truth, cmap='gray')
    ax[1].set_title('Ground Truth')
    plt.show()

if __name__ == '__main__':

    number_of_chucks_testset = 1

    # Load model
    model_path = MODEL_DIR
    model_name = 'model_20250219_161255_loss_30295.7045'
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path))
    criterion = WeightedCustomLoss()
    model.eval()

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    summary(model, (1, 400, 400))

    # Load scalers
    with open(os.path.join(SCALER_DIR, 'scaler_X.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(SCALER_DIR, 'scaler_y.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)

    # Shuffle files_lidar_1 and files_BB_1 in the same way
    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_1_TEST)]), sorted([f for f in os.listdir(POSITION_1_TEST)])))
    random.shuffle(combined_files)
    files_lidar_1, files_BB_1 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_1 = list(files_lidar_1)
    files_BB_1 = list(files_BB_1)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_1, POSITION_1_TEST)
    print(f"\nSum_complete_test_lidar1: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_test_lidar1: ", sum_ped/len(files_BB_1), sum_bic/len(files_BB_1), sum_car/len(files_BB_1))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_2_TEST)]), sorted([f for f in os.listdir(POSITION_2_TEST)])))
    random.shuffle(combined_files)
    files_lidar_2, files_BB_2 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_2 = list(files_lidar_2)
    files_BB_2 = list(files_BB_2)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_2, POSITION_2_TEST)
    print(f"\nSum_complete_test_lidar2: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_test_lidar2: ", sum_ped/len(files_BB_2), sum_bic/len(files_BB_2), sum_car/len(files_BB_2))

    combined_files = list(zip(sorted([f for f in os.listdir(LIDAR_3_TEST)]), sorted([f for f in os.listdir(POSITION_3_TEST)])))
    random.shuffle(combined_files)
    files_lidar_3, files_BB_3 = zip(*combined_files)
    # Convert back to lists if needed
    files_lidar_3 = list(files_lidar_3)
    files_BB_3 = list(files_BB_3)

    sum_ped, sum_bic, sum_car = number_of_BB(files_BB_3, POSITION_3_TEST)
    print(f"\nSum_complete_test_lidar3: ", sum_ped, sum_bic, sum_car)
    print(f"Average_complete_test_lidar3: ", sum_ped/len(files_BB_3), sum_bic/len(files_BB_3), sum_car/len(files_BB_3))

    # Total number of files for each lidar
    total_num_of_files1 = len(files_lidar_1)
    total_num_of_files2 = len(files_lidar_2)
    total_num_of_files3 = len(files_lidar_3)
    print(f"\nTotal number of files: {total_num_of_files1, total_num_of_files2, total_num_of_files3}")

    # Number of files of each chunck for each lidar
    file_for_chunck1 = math.ceil(total_num_of_files1 / number_of_chucks_testset) #type: ignore
    file_for_chunck2 = math.ceil(total_num_of_files2 / number_of_chucks_testset) #type: ignore
    file_for_chunck3 = math.ceil(total_num_of_files3 / number_of_chucks_testset) #type: ignore

    print(f"Number of files for each chunck: {file_for_chunck1, file_for_chunck2, file_for_chunck3}")

    gc.collect()

    test_loss = 0
    predictions = []

    i = 0

    for i in range(number_of_chucks_testset): #type: ignore
        
        complete_grid_maps = []
        complete_grid_maps_BB = []
        complete_numb_BB = []

        print(f"\nTest chunck number {i+1} of {number_of_chucks_testset}: ")

        files_lidar_chunck = files_lidar_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_lidar_1) ) ] #type: ignore
        files_BB_chunck = files_BB_1[ i*file_for_chunck1 : min( (i+1)*file_for_chunck1, len(files_BB_1) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_1_TEST, POSITION_1_TEST, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_1_TEST)
        print(f"\nSum_chunck_test_lidar1: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_test_lidar1: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))
        
        files_lidar_chunck = files_lidar_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_lidar_2) ) ] #type: ignore
        files_BB_chunck = files_BB_2[ i*file_for_chunck2 : min( (i+1)*file_for_chunck2, len(files_BB_2) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_2_TEST, POSITION_2_TEST, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False) # type: ignore
        
        # Info for lidar 2 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_2_TEST)
        print(f"\nSum_chunck_test_lidar2: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_test_lidar2: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        files_lidar_chunck = files_lidar_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_lidar_3) ) ] #type: ignore
        files_BB_chunck = files_BB_3[ i*file_for_chunck3 : min( (i+1)*file_for_chunck3, len(files_BB_3) ) ] #type: ignore
        generate_combined_grid_maps(LIDAR_3_TEST, POSITION_3_TEST, files_lidar_chunck, files_BB_chunck, complete_grid_maps, complete_grid_maps_BB, complete_numb_BB, False) # type: ignore

        # Info for lidar 1 about the number of bounding boxes
        sum_ped, sum_bic, sum_car = number_of_BB(files_BB_chunck, POSITION_3_TEST)
        print(f"\nSum_chunck_test_lidar3: ", sum_ped, sum_bic, sum_car)
        print(f"Average_chunck_test_lidar3: ", sum_ped/len(files_BB_chunck), sum_bic/len(files_BB_chunck), sum_car/len(files_BB_chunck))

        # Concatenate the lists in complete_grid_maps along the first dimension
        complete_grid_maps = np.array(complete_grid_maps)
        print(f"complete grid map shape : {complete_grid_maps.shape}")

        # Concatenate the lists in complete_grid_maps_BB along the first dimension
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)
        print(f"complete grid map BB shape : {complete_grid_maps_BB.shape}")

        #complete_num_BB = np.expand_dims(complete_num_BB, axis=1)
        #print(f"expanded number of bounding boxes shape : {complete_num_BB.shape}")

        gc.collect()

        # Normalize the data
        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        #complete_grid_maps_BB = scaler_y.transform(complete_grid_maps_BB.reshape(-1, complete_grid_maps_BB.shape[-1])).reshape(complete_grid_maps_BB.shape)

        print("\nShape after transform: ",complete_grid_maps.shape, complete_grid_maps_BB.shape)

        complete_grid_maps = np.expand_dims(complete_grid_maps, axis=1)
        complete_grid_maps_BB = np.expand_dims(complete_grid_maps_BB, axis=1)

        print("New shape test:", complete_grid_maps.shape, complete_grid_maps_BB.shape)
            
        # Prepare data loaders
        test_dataset = TensorDataset(torch.from_numpy(complete_grid_maps).float(), torch.from_numpy(complete_grid_maps_BB).float()) # Each element in train_dataset will be a tuple (input, target). Both will have shape (400,400). There will be as many elements in the dataset as there are samples in X_train.

        #del complete_grid_maps, complete_grid_maps_BB

        print("\nLenght dataset test:", len(test_dataset))

        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        del test_dataset
        gc.collect()

        # Make predictions
        with torch.no_grad():
            for data in test_loader:
                inputs, _ = data
                inputs= inputs.to(device)
                outputs = model(inputs)
                predictions.append(outputs)
        predictions = torch.cat(predictions).cpu().numpy()
        print("Predictions Shape:", predictions.shape)

    # Concatenate predictions
    predictions = np.concatenate(predictions, axis=0)

    print("Predictions Shape:", predictions.shape)

    print("Complete Grid Maps BB Shape:", complete_grid_maps_BB.shape)
    complete_grid_maps_BB = complete_grid_maps_BB.reshape(-1, 400, 400)
    print("Complete Grid Maps BB Shape:", complete_grid_maps_BB.shape)

    for i in range(complete_grid_maps_BB.shape[0]):
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        ax[0].imshow(complete_grid_maps_BB[i], cmap='gray')
        ax[0].set_title('Original Grid Map')
        
        ax[1].imshow(predictions[i], cmap='gray')
        ax[1].set_title(f'Prediction Grid Map')
        
        plt.show()
