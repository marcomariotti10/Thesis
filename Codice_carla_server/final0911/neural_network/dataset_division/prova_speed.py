import numpy as np
import os
from constants import *
from ffcv.fields import NDArrayField, FloatField
from ffcv.writer import DatasetWriter

class Dataset:
    def __init__(self, chunck_dir, i):
        self.X = np.load(os.path.join(chunck_dir, f'complete_grid_maps_{i}.npy'))  # Generate (N, 400, 400) arrays
        self.Y = np.load(os.path.join(chunck_dir, f'complete_grid_maps_BB_{i}.npy'))  # Generate (N, 400, 400) arrays

    def __getitem__(self, idx):
        return (self.X[idx].astype('float32'), self.Y[idx].astype('int'))

    def __len__(self):
        return len(self.X)
    
if __name__ == '__main__':

    shape = (400, 400)  # Shape of each sample

    if HOME[0] == 'C':
        chunck_dir = '/mnt/c' + CHUNCKS_DIR[2:]
        ffcv_dir = '/mnt/c' + FFCV_DIR[2:]
    else:
        chunck_dir = CHUNCKS_DIR
        ffcv_dir = FFCV_DIR
    

    for i in range (NUMBER_OF_CHUNCKS):

        dataset = Dataset(chunck_dir, i)

        x, y = dataset[0]
        print(f"X shape: {x.shape}, Y shape: {y.shape}")

        name = f"dataset{i}.beton"  # Define the path where the dataset will be written
        complete_path = os.path.join(ffcv_dir, name)

        writer = DatasetWriter(complete_path, {
            'covariate': NDArrayField(shape=shape, dtype=np.dtype('float32')),  # Adjust shape
            'label': NDArrayField(shape=shape, dtype=np.dtype('int')),
        }, num_workers=16)

        writer.from_indexed_dataset(dataset)

        #TODO: finish this, generate npy on server and use this then
