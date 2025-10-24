"""
FFCV Dataset Generation Module

This module converts NumPy array datasets into FFCV (Fast Forward Computer Vision) 
format for accelerated data loading during training. FFCV provides significant 
performance improvements over traditional data loaders by optimizing storage format 
and utilizing efficient decoding strategies.

Key features:
- Converts chunked NumPy datasets to FFCV .beton format
- Implements custom PyTorch Dataset for FFCV compatibility  
- Parallel data loading during conversion for efficiency
- Automatic cleanup of intermediate NumPy files
- Configurable data types and shapes for optimal storage

The conversion process:
1. Load NumPy data chunks in parallel
2. Create PyTorch Dataset wrapper with proper data types
3. Use FFCV DatasetWriter to convert to .beton format
4. Clean up intermediate files to save storage
5. Repeat for train, validation, and test sets

FFCV benefits:
- Faster data loading during training (up to 10x speedup)
- Reduced memory usage through optimized storage
- Built-in support for data augmentation pipelines
- Efficient GPU data transfer
"""

# Standard library imports
import sys
import os
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import numpy as np
import torch
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants and functions
from config import *

class DatasetNPY(torch.utils.data.Dataset):   
    """
    PyTorch Dataset wrapper for loading NumPy array data chunks.
    
    This custom dataset class loads pre-processed LiDAR grid data and bounding box
    targets from NumPy files, formatting them appropriately for FFCV conversion.
    It supports parallel loading for efficiency and ensures proper data types
    for optimal FFCV performance.
    
    The dataset loads:
    - Input data (X): LiDAR grid maps with multiple time steps/sensors
    - Target data (Y): Bounding box ground truth grids
    
    Data is loaded in parallel using ThreadPoolExecutor for improved I/O performance.
    """
    
    def __init__(self, name, i):
        """
        Initialize the dataset by loading data from NumPy files.
        
        Args:
            name (str): Dataset split name ('train', 'val', or 'test')
            i (int): Chunk index for this dataset split
            
        The constructor loads both input and target data in parallel using
        ThreadPoolExecutor to minimize I/O wait time.
        """
        # Construct the path to the chunk data directory
        complete_name_chunck_path = os.path.join(FFCV_DIRECTORY)

        # Load input and target data in parallel for efficiency
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Load both grid maps (input) and bounding box targets (output) simultaneously
            self.X, self.Y = executor.map(load_array, [
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_{name}_{i}.npy'),      # Input: LiDAR grid data
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_{name}_{i}.npy')   # Target: Bounding box data
            ])

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (input_data, target_data) with appropriate data types
                - input_data (float32): LiDAR grid map data
                - target_data (uint8): Bounding box target data
                
        Data types are optimized for FFCV performance:
        - float32 for continuous input data (sufficient precision, smaller size)
        - uint8 for binary/categorical target data (minimal storage)
        """
        return (self.X[idx].astype('float32'), self.Y[idx].astype('uint8'))

    def __len__(self):
        """
        Get the total number of samples in the dataset.
        
        Returns:
            int: Number of samples in the input data array
        """
        return len(self.X)
    
def load_dataset_NPY(name, i):
    """
    Convert a NumPy dataset chunk to FFCV format (.beton file).
    
    This function handles the complete conversion pipeline from NumPy arrays
    to FFCV format, including dataset creation, format specification, and
    file cleanup. FFCV format provides significant performance improvements
    for data loading during model training.
    
    Args:
        name (str): Dataset split name ('train', 'val', or 'test')
        i (int): Chunk index for this dataset split
        
    Returns:
        None: Creates .beton file and removes source NumPy files
        
    Process:
        1. Create PyTorch dataset wrapper for the NumPy data
        2. Display data shapes for verification
        3. Define FFCV field specifications for input and target data
        4. Use DatasetWriter to convert to .beton format
        5. Clean up source NumPy files to save storage
        
    FFCV Optimization:
        - Uses NDArrayField for efficient array storage
        - Specifies optimal data types (float32 for inputs, uint8 for targets)
        - Configures multi-threaded writing for performance
        - Compresses data for reduced storage footprint
    """
    # Create PyTorch dataset wrapper for the NumPy data
    dataset = DatasetNPY(name, i)
    
    # Get sample data to verify shapes and types
    x, y = dataset[0]

    # Generate output filename for the FFCV dataset
    new_name = f"dataset_{name}{i}.beton"  # .beton is FFCV's optimized format

    # Construct full output path
    complete_name_ffcv_path = os.path.join(FFCV_DIRECTORY)
    complete_path = os.path.join(complete_name_ffcv_path, new_name)

    # Create FFCV DatasetWriter with field specifications
    writer = DatasetWriter(complete_path, {
            # Input data: LiDAR grid maps with multiple time steps
            'covariate': NDArrayField(shape=shape_input, dtype=np.dtype('float32')),  # Continuous input data
            # Target data: Bounding box binary grids  
            'label': NDArrayField(shape=shape_target, dtype=np.dtype('uint8')),       # Binary/categorical targets
        }, num_workers=16)  # Use 16 workers for parallel writing

    # Convert the PyTorch dataset to FFCV format
    writer.from_indexed_dataset(dataset)

    # Clean up source NumPy files to save storage space
    complete_name_chunck_path = os.path.join(FFCV_DIRECTORY)
    
    # Remove input data file
    os.remove(os.path.join(complete_name_chunck_path, f'complete_grid_maps_{name}_{i}.npy'))
    # Remove target data file
    os.remove(os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_{name}_{i}.npy'))

# Main execution block
if __name__ == '__main__':
    
    # Define data shapes for FFCV field specifications
    shape_input = (NUMBER_FRAMES_INPUT, 400, 400)     # Shape of input LiDAR grid sequences
    shape_target = (len(FUTURE_TARGET_RILEVATION), 400, 400)  # Shape of target bounding box grids

    # Ensure output directory exists
    complete_name_ffcv_path = os.path.join(FFCV_DIRECTORY)
    os.makedirs(complete_name_ffcv_path, exist_ok=True)

    # Convert all training dataset chunks to FFCV format
    print("Converting training chunks to FFCV format...")
    for i in range(NUMBER_OF_CHUNCKS_TRAIN):
        print(f"Processing training chunk {i+1}/{NUMBER_OF_CHUNCKS_TRAIN}")
        load_dataset_NPY('train', i)

    # Convert all test dataset chunks to FFCV format  
    print("\nConverting test chunks to FFCV format...")
    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"Processing test chunk {i+1}/{NUMBER_OF_CHUNCKS_TEST}")
        load_dataset_NPY('test', i)

    # Convert all validation dataset chunks to FFCV format
    print("\nConverting validation chunks to FFCV format...")
    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"Processing validation chunk {i+1}/{NUMBER_OF_CHUNCKS_TEST}")
        load_dataset_NPY('val', i)
    
    print("\nFFCV conversion completed! All .beton files are ready for training.")