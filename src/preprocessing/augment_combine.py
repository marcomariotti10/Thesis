"""
Data Augmentation Combination Module

This module combines original training data with augmented versions to increase dataset
diversity and improve model robustness. It implements a randomized combination strategy
where different chunks of original data are paired with different chunks of augmented
data to maximize variation while maintaining data balance.

Key features:
- Random pairing of original and augmented data chunks
- Memory-efficient parallel loading of data arrays
- In-place data shuffling to prevent overfitting patterns
- Automatic cleanup of temporary augmented files
- Deterministic randomization using fixed seed

The combination process:
1. Generate random pairing lists for original and augmented chunks
2. For each chunk pair:
   a. Load original and augmented data in parallel
   b. Concatenate the arrays
   c. Shuffle the combined data
   d. Save the enhanced training chunk
   e. Clean up temporary augmented files

This approach ensures that the model sees both original and augmented versions
of the data while preventing memorization of specific augmentation patterns.
"""

# Standard library imports
import sys
import os
import random
import gc
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import numpy as np

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants and functions
from config import *

# Main execution block
if __name__ == '__main__':
    
    # Trigger garbage collection to free up memory before processing
    gc.collect()

    # Set random seed for reproducible chunk pairing
    random.seed(SEED)

    # Set up the directory path for chunk files
    complete_name_chunck_path = os.path.join(FFCV_DIRECTORY)

    # Ensure the output directory exists
    os.makedirs(complete_name_chunck_path, exist_ok=True)

    # Generate random pairing lists for combining original and augmented chunks
    train_list = random.sample(range(NUMBER_OF_CHUNCKS_TRAIN), NUMBER_OF_CHUNCKS_TRAIN)     # Original training chunks
    augment_list = random.sample(range(NUMBER_OF_CHUNCKS_TRAIN), NUMBER_OF_CHUNCKS_TRAIN)   # Augmented chunks to pair

    # Process each chunk combination
    for i in range(NUMBER_OF_CHUNCKS_TRAIN):
        
        # Get the specific chunk indices for this iteration
        train_chunck = train_list[i]      # Original training chunk to use
        augment_chunck = augment_list[i]  # Augmented chunk to combine with

        # Load data arrays in parallel for efficiency
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Load original and augmented data: input grids and bounding box targets
            complete_grid_maps, complete_grid_maps_BB, augmented_grid_maps, augmented_grid_maps_BB = executor.map(load_array, [
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_train_{train_chunck}.npy'),        # Original input data
                os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_train_{train_chunck}.npy'),     # Original target data  
                os.path.join(complete_name_chunck_path, f'augmented_grid_maps_train_{augment_chunck}.npy'),     # Augmented input data
                os.path.join(complete_name_chunck_path, f'augmented_grid_maps_BB_train_{augment_chunck}.npy')   # Augmented target data
            ])

        # Combine original and augmented data by concatenating along the first axis
        complete_grid_maps = np.concatenate((complete_grid_maps, augmented_grid_maps), axis=0)
        complete_grid_maps_BB = np.concatenate((complete_grid_maps_BB, augmented_grid_maps_BB), axis=0)

        # Shuffle the combined data to prevent learning augmentation patterns
        indices = np.arange(complete_grid_maps.shape[0])
        np.random.shuffle(indices)
        complete_grid_maps = complete_grid_maps[indices]
        complete_grid_maps_BB = complete_grid_maps_BB[indices]

        # Save the enhanced training chunk back to disk
        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_train_{train_chunck}.npy'), complete_grid_maps)
        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_train_{train_chunck}.npy'), complete_grid_maps_BB)

        # Clean up temporary augmented files to save storage space
        os.remove(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_train_{augment_chunck}.npy'))
        os.remove(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_BB_train_{augment_chunck}.npy'))