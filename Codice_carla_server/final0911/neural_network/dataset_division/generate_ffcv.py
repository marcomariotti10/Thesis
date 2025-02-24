import numpy as np
import os
import sys
from ffcv.fields import NDArrayField, FloatField
import torch
import torch.utils.data
from ffcv.writer import DatasetWriter

class DatasetNPY(torch.utils.data.Dataset):   
    def __init__(self, name, i):
        with ThreadPoolExecutor(max_workers=2) as executor:
            self.X , self.Y = executor.map(load_array, [
            os.path.join(CHUNCKS_DIR, f'complete_grid_maps_{name}_{i}.npy'),
            os.path.join(CHUNCKS_DIR, f'complete_grid_maps_BB_{name}_{i}.npy')
        ])

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx].astype('int'))

    def __len__(self):
        return len(self.X)
    
def load_dataset(name, i):
    dataset = DatasetNPY(name, i)
    x, y = dataset[0]
    print(f"X shape dataset {name}: {x.shape}, Y shape dataset {name}: {y.shape}")

    new_name = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_path = os.path.join(FFCV_DIR, new_name)

    writer = DatasetWriter(complete_path, {
            'covariate': NDArrayField(shape=shape, dtype=np.dtype('float32')),  # Adjust shape
            'label': NDArrayField(shape=shape, dtype=np.dtype('int')),
        }, num_workers=16)

    writer.from_indexed_dataset(dataset)

if __name__ == '__main__':

    shape = (1,400, 400)  # Shape of each sample

    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *
    
    os.makedirs(FFCV_DIR, exist_ok=True)

    for i in range (NUMBER_OF_CHUNCKS):

        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(load_dataset, 'train', i)
            executor.submit(load_dataset, 'val', i)

    i = 0

    for i in range (NUMBER_OF_CHUNCKS_TEST):

        load_dataset('test', i)