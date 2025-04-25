import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import numpy as np
import os
import sys
from ffcv.fields import NDArrayField, FloatField
import torch
import torch.utils.data
from ffcv.writer import DatasetWriter

class DatasetNPY(torch.utils.data.Dataset):   
    def __init__(self, name, i):
        
        complete_name_chunck_path = os.path.join(CHUNCKS_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')

        with ThreadPoolExecutor(max_workers=2) as executor:
            self.X , self.Y = executor.map(load_array, [
            os.path.join(complete_name_chunck_path, f'complete_grid_maps_{name}_{i}.npy'),
            os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_{name}_{i}.npy')
        ])

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx].astype('int'))

    def __len__(self):
        return len(self.X)
    
def load_dataset_NPY(name, i):
    gc.collect()
    
    dataset = DatasetNPY(name, i)
    x, y = dataset[0]
    print(f"X shape dataset {name}: {x.shape}, Y shape dataset {name}: {y.shape}")

    new_name = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written

    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path = os.path.join(complete_name_ffcv_path, new_name)

    writer = DatasetWriter(complete_path, {
            'covariate': NDArrayField(shape=shape_input, dtype=np.dtype('float32')),  # Adjust shape
            'label': NDArrayField(shape=shape_target, dtype=np.dtype('int')),
        }, num_workers=16, page_size=1024*1024*16)

    writer.from_indexed_dataset(dataset)

    complete_name_chunck_path = os.path.join(CHUNCKS_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')

    os.remove(os.path.join(complete_name_chunck_path, f'complete_grid_maps_{name}_{i}.npy'))
    os.remove(os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_{name}_{i}.npy'))

if __name__ == '__main__':

    shape_input = (NUMBER_RILEVATIONS_INPUT,400,400)  # Shape of each sample
    shape_target = (len(FUTURE_TARGET_RILEVATION),400,400)  # Shape of each target

    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    
    os.makedirs(complete_name_ffcv_path, exist_ok=True)

    for i in range (NUMBER_OF_CHUNCKS):

        load_dataset_NPY('train', i)

    i = 0

    for i in range (NUMBER_OF_CHUNCKS_TEST):

        load_dataset_NPY('test', i)

    i = 0

    for i in range (NUMBER_OF_CHUNCKS_TEST):
        
        load_dataset_NPY('val', i)