import numpy as np
import os
import sys
from ffcv.fields import NDArrayField, FloatField
import torch
import torch.utils.data
from ffcv.writer import DatasetWriter

class DatasetNPY(torch.utils.data.Dataset):   
    def __init__(self, chunck_dir, i, bool):
        if bool:
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.X , self.Y = executor.map(load_array, [
                os.path.join(chunck_dir, f'complete_grid_maps_{i}.npy'),
                os.path.join(chunck_dir, f'complete_grid_maps_BB_{i}.npy')
            ])
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.X , self.Y = executor.map(load_array, [
                os.path.join(chunck_dir, f'complete_grid_maps_test_{i}.npy'),
                os.path.join(chunck_dir, f'complete_grid_maps_BB_test_{i}.npy')
            ])

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx].astype('int'))

    def __len__(self):
        return len(self.X)
    
if __name__ == '__main__':

    shape = (400, 400)  # Shape of each sample

    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *

    if HOME[0] == 'C':
        chunck_dir = '/mnt/c' + CHUNCKS_DIR[2:]
        ffcv_dir = '/mnt/c' + FFCV_DIR[2:]
    else:
        chunck_dir = CHUNCKS_DIR
        ffcv_dir = FFCV_DIR
    
    os.makedirs(ffcv_dir, exist_ok=True)

    for i in range (NUMBER_OF_CHUNCKS):

        dataset = DatasetNPY(chunck_dir, i, True)

        x, y = dataset[0]
        print(f"X shape: {x.shape}, Y shape: {y.shape}")

        name = f"dataset{i}.beton"  # Define the path where the dataset will be written
        complete_path = os.path.join(ffcv_dir, name)

        writer = DatasetWriter(complete_path, {
            'covariate': NDArrayField(shape=shape, dtype=np.dtype('float32')),  # Adjust shape
            'label': NDArrayField(shape=shape, dtype=np.dtype('int')),
        }, num_workers=16)

        writer.from_indexed_dataset(dataset)

    for i in range (NUMBER_OF_CHUNCKS_TEST):

        dataset = DatasetNPY(chunck_dir, i, False)

        x, y = dataset[0]
        print(f"X shape: {x.shape}, Y shape: {y.shape}")

        name = f"dataset_test{i}.beton"

        complete_path = os.path.join(ffcv_dir, name)

        writer = DatasetWriter(complete_path, {
            'covariate': NDArrayField(shape=shape, dtype=np.dtype('float32')),  # Adjust shape
            'label': NDArrayField(shape=shape, dtype=np.dtype('int')),
        }, num_workers=16)

        writer.from_indexed_dataset(dataset)