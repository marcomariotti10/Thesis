"""
Dataset Division and Augmentation Module

This module implements the core dataset preparation pipeline that divides multi-sensor
LiDAR data into training, validation, and test sets while applying data augmentation
to enhance model robustness. It processes large datasets in chunks to manage memory
efficiently and applies normalization using pre-fitted scalers.

Key features:
- Multi-sensor data integration and synchronization
- Chunked processing for memory-efficient handling of large datasets
- Data augmentation on training samples to increase diversity
- Automatic train/validation/test split generation
- Normalization using pre-fitted MinMaxScaler
- Statistical analysis of bounding box distributions
- Memory management with garbage collection

The pipeline workflow:
1. Load and synchronize LiDAR and bounding box data from multiple sensors
2. Generate combined file lists for temporal alignment
3. Process data in chunks to manage memory constraints
4. Apply normalization using pre-fitted scalers
5. For training data: apply augmentation to selected samples
6. Save processed chunks as NumPy arrays for further processing

This module serves as the bridge between raw sensor data and the final
FFCV format used for efficient model training.
"""

# Standard library imports
import sys
import os
import pickle
import random
import math
import gc
import cv2
from multiprocessing import set_start_method

# Third-party imports
import numpy as np

# Add the parent directory to the Python path to access config module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration constants and functions
from config import *

def generate_combined_grid_maps_pred(grid_map_path, grid_map_BB_path, grid_map_files, complete_grid_maps, complete_grid_maps_BB):
        
    for i in range(len(grid_map_files)):

        actors = {}
        
        grid_map_group = []

        for j in range (NUMBER_FRAMES_INPUT):

            complete_path = os.path.join(grid_map_path, grid_map_files[i][j])

            points = load_points_grid_map(complete_path)

            grid_map_recreate = np.full((DIMENSION_Y, DIMENSION_X), NO_DATA_VALUE, dtype=float) # type: ignore

            cols, rows, heights = points.T
            grid_map_recreate[rows.astype(int), cols.astype(int)] = heights.astype(float)

            grid_map_group.append(grid_map_recreate)

            complete_path_BB = os.path.join(grid_map_BB_path, grid_map_files[i][NUMBER_FRAMES_INPUT + j])

            with open(complete_path_BB, 'r') as file_BB:
                reader = csv.reader(file_BB)
                next(reader)  # Skip header
                for row in reader: 
                    if row[3] == 1:
                        if row[0] not in actors:
                            actors[row[0]] = 1
                        else:
                            actors[row[0]] += 1

        complete_grid_maps.append(grid_map_group)

        grid_map_BB_group = []

        #Save the input
        for k in range(len(FUTURE_TARGET_RILEVATION)):   

            complete_path_BB = os.path.join(grid_map_BB_path, grid_map_files[i][-len(FUTURE_TARGET_RILEVATION) + k])

            points_BB, indeces = load_points_grid_map_BB(complete_path_BB)

            all_pairs = []

            # Iterate through each row in the numpy array
            for idx, row in enumerate(points_BB):
                if indeces[idx] in actors and actors[indeces[idx]] >= MINIMUM_NUMBER_OF_DETECTIONS:
                    # Extract the string from the array
                    string_data = row[0]
                    # Safely evaluate the string to convert it into a list of tuples
                    pairs = ast.literal_eval(string_data)
                    # Add the pairs to the all_pairs list
                    all_pairs.extend(pairs)
                
            all_pairs = np.array(all_pairs)

            grid_map_recreate_BB = np.full((DIMENSION_Y, DIMENSION_X), 0, dtype=float) # type: ignore

            if len(all_pairs) != 0:
                cols, rows = all_pairs.T
                grid_map_recreate_BB[rows.astype(int), cols.astype(int)] = 1

            grid_map_BB_group.append(grid_map_recreate_BB)

        complete_grid_maps_BB.append(grid_map_BB_group)

def generate_combined_list(files_lidar, files_BB, type):
    """
    Generate a new list where each element contains NUMBER_FRAMES_INPUT elements from files_lidar
    and 1 element from files_BB, which is FUTURE_TARGET_RILEVATION positions ahead of the last lidar element.

    Parameters:
    - files_lidar: List of lists containing sorted lidar file names.
    - files_BB: List of lists containing sorted bounding box file names.
    - NUMBER_FRAMES_INPUT: Number of lidar files to include in each group.
    - FUTURE_TARGET_RILEVATION: Number of positions ahead for the bounding box file.

    Returns:
    - combined_list: A list where each element contains NUMBER_FRAMES_INPUT lidar files and 1 bounding box file.
    """
    combined_list = []
    #Can't use match because not supported by python 3.9
    if type == 'train':
        start =  int(len(files_lidar)*TEST_SIZE_PERCENTAGE)
        finish = int(len(files_lidar)*TEST_SIZE_PERCENTAGE*9)
    elif type == 'val':
        start = int(len(files_lidar)*TEST_SIZE_PERCENTAGE*9)
        finish = int(len(files_lidar))
    elif type == 'test':
        start = 0
        finish = int(len(files_lidar)*TEST_SIZE_PERCENTAGE)

    for i in range(start, finish - NUMBER_FRAMES_INPUT - FUTURE_TARGET_RILEVATION[-1] + 1, NUMBER_FRAMES_INPUT):
        # Take NUMBER_FRAMES_INPUT lidar files starting from the current index
        lidar_group = files_lidar[i:i + NUMBER_FRAMES_INPUT]
        # Take the FUTURE_TARGET_RILEVATION-th BB file after the last lidar file
        BB_files = files_BB[i:i + NUMBER_FRAMES_INPUT]
        BB_file_future = []
        for j in range(len(FUTURE_TARGET_RILEVATION)):
            BB_file_future.append(files_BB[i + NUMBER_FRAMES_INPUT + FUTURE_TARGET_RILEVATION[j] - 1])
        # Combine the lidar group and the BB file into one element
        combined_list.append(lidar_group + BB_files + BB_file_future)

    random.shuffle(combined_list)
    return combined_list

def apply_augmentation(random_gm, random_BB):
    """
    Apply data augmentation techniques to randomly selected LiDAR and bounding box samples.
    
    This function implements a comprehensive augmentation pipeline that applies multiple
    spatial transformations to increase dataset diversity and improve model robustness.
    It uses a probabilistic approach to select and combine different augmentation techniques
    while ensuring consistent transformations across input and target data.
    
    Args:
        random_gm (np.ndarray): Random subset of LiDAR grid maps for augmentation
                               Shape: (n_samples, n_timeframes, height, width)
        random_BB (np.ndarray): Corresponding bounding box grid maps
                               Shape: (n_samples, n_future_steps, height, width)
    
    Returns:
        tuple: (augmented_grid_maps, augmented_grid_maps_BB)
            - augmented_grid_maps (list): Augmented LiDAR grid maps
            - augmented_grid_maps_BB (list): Augmented bounding box grid maps
    
    Augmentation Techniques:
        1. Rotation: Random rotation between -60° to -30° or 30° to 60°
        2. Spatial Shift: Random translation along vertical or horizontal axis
        3. Flip: Random vertical or horizontal mirroring
    
    Augmentation Strategy:
        - Each sample receives exactly 2 different augmentation techniques
        - Probabilistic selection ensures variety in augmentation combinations
        - Same transformations applied to both input and target data
        - Maintains spatial consistency between LiDAR and bounding box data
    """
    
    def random_shift(img, axis, shift):
        """
        Apply a shift to the image along the specified axis and pad with zeros.
        
        Args:
            img (np.ndarray): Input image to shift
            axis (int): Axis along which to shift (0=vertical, 1=horizontal)  
            shift (int): Number of pixels to shift (positive/negative for direction)
        
        Returns:
            np.ndarray: Shifted image with zero-padded regions
            
        Shift Behavior:
            - Positive shift: Move content right (horizontal) or down (vertical)
            - Negative shift: Move content left (horizontal) or up (vertical)
            - Exposed areas filled with zeros to maintain image dimensions
        """
        shifted_img = np.roll(img, shift=shift, axis=axis)
        if axis == 1:  # Horizontal shift
            if shift > 0:  # Positive shift (right)
                shifted_img[:, :shift] = 0
            else:  # Negative shift (left)
                shifted_img[:, shift:] = 0
        elif axis == 0:  # Vertical shift
            if shift > 0:  # Positive shift (down)
                shifted_img[:shift, :] = 0
            else:  # Negative shift (up)
                shifted_img[shift:, :] = 0
        return shifted_img

    def random_rotation(img, angle):
        """
        Rotate an image by a specified angle.
        
        Args:
            img (np.ndarray): Input image to rotate
            angle (float): Rotation angle in degrees (positive=counterclockwise)
        
        Returns:
            np.ndarray: Rotated image with preserved dimensions
            
        Note:
            Uses the rotate_image function from config module for consistent
            rotation behavior across the entire pipeline.
        """
        return rotate_image(img, angle)

    # Initialize lists to store augmented samples
    augmented_grid_maps = []     # Augmented LiDAR grid maps
    augmented_grid_maps_BB = []  # Augmented bounding box grid maps

    # Set the random seed for consistent and reproducible augmentation
    random.seed(SEED)
    
    # Process each sample in the random subset
    for i in range(random_gm.shape[0]):
        # Create deep copies to avoid modifying the original arrays
        # This preserves the original data for potential future use
        grid_map = np.copy(random_gm[i])
        grid_map_BB = np.copy(random_BB[i])

        # Define augmentation probabilities for technique selection
        # Each technique has a different probability to encourage diversity
        augmentations = [
            ('rotation', ROTATION_PROBABILITY),  # 50% weight - moderate rotational variation
            ('shift', SHIFT_PROBABILITY),     # 20% weight - less frequent spatial shifts
            ('flip', FLIP_PROBABILITY)       # 30% weight - common mirroring operations
        ]

        # Select the first augmentation using weighted random selection
        # This ensures probabilistic diversity in augmentation combinations
        first_augmentation = random.choices(
            augmentations, 
            weights=[aug[1] for aug in augmentations], 
            k=1
        )[0]

        # Remove the first augmentation from the list for the second selection
        # This guarantees that each sample gets exactly 2 different augmentation types
        remaining_augmentations = [aug for aug in augmentations if aug[0] != first_augmentation[0]]

        # Select the second augmentation from the remaining techniques
        # This ensures each sample gets exactly 2 different augmentation types
        second_augmentation = random.choices(
            remaining_augmentations, 
            weights=[aug[1] for aug in remaining_augmentations], 
            k=1
        )[0]

        # Apply the selected augmentations in sequence
        for augmentation_type in [first_augmentation, second_augmentation]:
            if augmentation_type[0] == 'rotation':
                # Random rotation with significant angles to simulate orientation changes
                # Bimodal distribution: either negative (-60° to -30°) or positive (30° to 60°)
                angle = int(random.uniform(-MAX_ROTATION_ANGLE, -MINIMUM_ROTATION_ANGLE)) if random.random() < 0.5 else int(random.uniform(MINIMUM_ROTATION_ANGLE, MAX_ROTATION_ANGLE))
                augmentation = lambda img: random_rotation(img, angle=angle)
                
            elif augmentation_type[0] == 'shift':
                # Random spatial shift to simulate position variations
                # Bimodal distribution: either negative or positive displacement
                shift = int(random.uniform(-MAX_SHIFT, -MINIMUM_SHIFT)) if random.random() < 0.5 else int(random.uniform(MINIMUM_SHIFT, MAX_SHIFT))
                
                # Randomly choose shift direction (0=vertical, 1=horizontal)
                axis = random.choice([0, 1])
                augmentation = lambda img: random_shift(img, axis=axis, shift=shift)
                
            elif augmentation_type[0] == 'flip':
                # Random mirroring to simulate different viewpoints
                # 0 = vertical flip (upside down), 1 = horizontal flip (left-right)
                flip_code = random.choice([0, 1])
                augmentation = lambda img: cv2.flip(img, flip_code)

            # Apply the same augmentation to all timeframes in the LiDAR sequence
            for k in range(grid_map.shape[0]):
                grid_map[k] = augmentation(grid_map[k])
                
            # Apply the same augmentation to all future timeframes in the bounding box sequence
            for z in range(grid_map_BB.shape[0]):
                grid_map_BB[z] = augmentation(grid_map_BB[z])

        # Add the augmented sample to the result collections
        augmented_grid_maps.append(grid_map)
        augmented_grid_maps_BB.append(grid_map_BB)

    return augmented_grid_maps, augmented_grid_maps_BB

def rotate_image(image, angle):
    """
    Rotate an image by a specified angle.
    """
    # Get the center of the image
    center = (image.shape[1] // 2, image.shape[0] // 2)
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated_image


def process_lidar_chunk(lidar_directory, position_directory, files_lidar_chunck, complete_grid_maps, complete_grid_maps_BB):
    """
    Process a chunk of LiDAR and position files from a single sensor.
    
    This function loads and processes a subset of temporally aligned LiDAR grid maps
    and corresponding bounding box data from one sensor. It serves as a wrapper
    around the grid map generation function to enable modular processing.
    
    Args:
        lidar_directory (str): Path to the directory containing LiDAR grid files
        position_directory (str): Path to the directory containing bounding box files
        files_lidar_chunck (list): List of temporally aligned file pairs to process
        complete_grid_maps (list): List to accumulate processed LiDAR grid maps
        complete_grid_maps_BB (list): List to accumulate processed bounding box grids
    
    Returns:
        tuple: (complete_grid_maps, complete_grid_maps_BB) with new data appended
    """
    # Generate combined grid maps for this sensor's chunk
    generate_combined_grid_maps_pred(lidar_directory, position_directory, files_lidar_chunck, complete_grid_maps, complete_grid_maps_BB) # type: ignore

    return complete_grid_maps, complete_grid_maps_BB

def generate_chunk(lidar_paths, position_paths, num_chunks, chunk_type):
    """
    Generate dataset chunks for a specific split (train/val/test) from multi-sensor data.
    
    This function implements the core dataset preparation pipeline that processes
    multi-sensor LiDAR and bounding box data, divides it into manageable chunks,
    applies normalization and augmentation, and saves the results for training.
    
    Args:
        lidar_paths (list): List of directory paths for LiDAR grid data from each sensor
        position_paths (list): List of directory paths for bounding box data from each sensor
        num_chunks (int): Number of chunks to divide the dataset into
        chunk_type (str): Type of dataset split ('train', 'val', or 'test')
    
    Returns:
        None: Saves processed chunks as NumPy arrays to disk
        
    Process:
        1. Generate temporally aligned file lists for all sensors
        2. Analyze bounding box statistics for dataset understanding
        3. Calculate optimal chunk sizes based on total data and memory constraints
        4. For each chunk:
           a. Load and process data from all sensors
           b. Apply normalization using pre-fitted scalers
           c. For training data: apply augmentation to increase diversity
           d. Save processed data as NumPy arrays
    """
    
    # Initialize lists to store file information for each sensor
    all_files = []         # Temporally aligned file pairs for each sensor
    files_for_chunck = []  # Number of files per chunk for each sensor

    # Process each sensor to generate aligned file lists and analyze statistics
    for i in range(NUMBER_OF_SENSORS):
        # Generate temporally aligned file pairs (LiDAR + bounding box)
        # This ensures that each sample contains synchronized sensor and annotation data
        all_files.append(generate_combined_list(sorted([f for f in os.listdir(lidar_paths[i])]),sorted([f for f in os.listdir(position_paths[i])]), chunk_type))

    
    # Calculate chunk sizes for each sensor based on total files and desired chunk count
    for i in range(NUMBER_OF_SENSORS):
        # Use ceiling division to ensure all files are included in chunks
        files_for_chunck.append(math.ceil(len(all_files[i]) / num_chunks)) #type: ignore
    
    # Trigger garbage collection to free memory before processing
    gc.collect()

    # Process each chunk sequentially
    for i in range(num_chunks):

        # Initialize lists to accumulate data from all sensors for this chunk
        complete_grid_maps = []     # LiDAR grid maps (input data)
        complete_grid_maps_BB = []  # Bounding box grids (target data)

        # Prepare file subsets for each sensor for this chunk
        all_files_chunck = []
        for k in range(NUMBER_OF_SENSORS):
            # Calculate start and end indices for this chunk
            start_idx = i * files_for_chunck[k]
            end_idx = min((i+1) * files_for_chunck[k], len(all_files[k]))
            
            # Extract file subset for this sensor and chunk
            files_lidar_chunck = all_files[k][start_idx:end_idx] #type: ignore
            all_files_chunck.append(files_lidar_chunck)
        
        # Process each sensor's chunk sequentially
        for j in range(NUMBER_OF_SENSORS): 
            process_lidar_chunk(lidar_paths[j], position_paths[j], all_files_chunck[j], complete_grid_maps, complete_grid_maps_BB)

        
        # Convert lists to numpy arrays for efficient processing
        complete_grid_maps = np.array(complete_grid_maps)
        complete_grid_maps_BB = np.array(complete_grid_maps_BB)

        # Apply normalization using pre-fitted scaler
        complete_grid_maps = scaler_X.transform(complete_grid_maps.reshape(-1, complete_grid_maps.shape[-1])).reshape(complete_grid_maps.shape)

        # For validation and test sets: apply random shuffling to prevent ordering bias
        if (chunk_type == 'val' or chunk_type == 'test'):
            indices = np.arange(complete_grid_maps.shape[0])
            np.random.shuffle(indices)
            complete_grid_maps = complete_grid_maps[indices]
            complete_grid_maps_BB = complete_grid_maps_BB[indices]

        # Display chunk information for verification

        # Apply data augmentation only to training data
        if(chunk_type =='train'):
            # Calculate number of samples to augment based on augmentation factor
            num_samples = int(complete_grid_maps.shape[0] * AUGMENTATION_PERCENTAGE)

            # Randomly select samples for augmentation (without replacement)
            random_indices = np.random.choice(complete_grid_maps.shape[0], num_samples, replace=False)

            # Extract selected samples for augmentation
            random_complete_grid_maps = complete_grid_maps[random_indices]
            random_complete_grid_maps_BB = complete_grid_maps_BB[random_indices]

            # Apply augmentation transformations (spatial, temporal, etc.)
            augmented_grid_maps, augmented_grid_maps_BB = apply_augmentation(random_complete_grid_maps, random_complete_grid_maps_BB)

            # Free memory by deleting intermediate variables
            del random_complete_grid_maps, random_complete_grid_maps_BB
            
            # Convert augmented data to numpy arrays
            augmented_grid_maps = np.array(augmented_grid_maps)
            augmented_grid_maps_BB = np.array(augmented_grid_maps_BB)


            # Save augmented data arrays for later combination with original data
            np.save(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_train_{i}.npy'), augmented_grid_maps)
            np.save(os.path.join(complete_name_chunck_path, f'augmented_grid_maps_BB_train_{i}.npy'), augmented_grid_maps_BB)

        # Save the processed chunk data
        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_{chunk_type}_{i}.npy'), complete_grid_maps)
        np.save(os.path.join(complete_name_chunck_path, f'complete_grid_maps_BB_{chunk_type}_{i}.npy'), complete_grid_maps_BB)
        print(f"shard {i+1} completed")

# Main execution block
if __name__ == '__main__':
    
    # Initialize memory management and multiprocessing
    gc.collect()  # Free up memory before starting intensive processing
    #set_start_method("spawn", force=True)  # Set multiprocessing method for compatibility
    
    # Set random seed for reproducible data shuffling and augmentation
    random.seed(SEED)

    # Set up output directory for processed chunks
    complete_name_chunck_path = os.path.join(FFCV_DIRECTORY)
    os.makedirs(complete_name_chunck_path, exist_ok=True)

    # Load pre-fitted normalization scaler for consistent data preprocessing
    # This scaler was fitted on the entire dataset to ensure proper normalization
    with open(os.path.join(SCALER_DIR, 'scaler.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)

    # Build directory paths for all sensors
    lidar_direcory_list = []  # Paths to LiDAR grid data for each sensor
    BB_directory_list = []    # Paths to bounding box data for each sensor

    # Create sensor-specific directory paths by replacing 'X' placeholder
    for i in range(1, NUMBER_OF_SENSORS+1):
        lidar_direcory_list.append(HEIGHTMAP_DIRECTORY.replace('X', str(i)))
        BB_directory_list.append(OGM_DIRECTORY.replace('X', str(i)))
    
    # Generate training dataset chunks with augmentation
    print("GENERATING TRAINING DATASET")
    generate_chunk(lidar_direcory_list, BB_directory_list, NUMBER_OF_CHUNCKS_TRAIN, 'train')

    # Generate test dataset chunks (no augmentation)
    print("GENERATING TEST DATASET")
    generate_chunk(lidar_direcory_list, BB_directory_list, NUMBER_OF_CHUNCKS_TEST, 'test')

    # Generate validation dataset chunks (no augmentation)
    print("GENERATING VALIDATION DATASET")
    generate_chunk(lidar_direcory_list, BB_directory_list, NUMBER_OF_CHUNCKS_TEST, 'val')
    
    print("DATASET PREPARATION COMPLETED")