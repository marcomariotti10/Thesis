import sys
import os
import gc
import ast
import random
import math
import torch
import shutil
import platform
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from datetime import datetime
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import KFold
from multiprocessing import set_start_method

if platform.system() == 'Linux':
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, ToDevice

K_FOLDS = 1
TEST_SIZE = 0.1  # 10%
VAL_SIZE = 0.1   # 10%
SEED = 42

def process_grid_maps_streaming(file_list, ipca, i, batch_size=32):
    batch = []
    for file in tqdm(file_list, desc=f"Sensor {i}"):
        complete_path = SNAPSHOT_X_GRID_DIRECTORY.replace('X', str(i)) + '/' + file
        points_BB, _ = load_points_grid_map_BB(complete_path)

        all_pairs = []
        for row in points_BB:
            string_data = row[0]
            pairs = ast.literal_eval(string_data)
            all_pairs.extend(pairs)

        all_pairs = np.array(all_pairs)
        grid = np.zeros((Y_RANGE, X_RANGE), dtype=np.float32)
        if len(all_pairs) > 0:
            cols, rows = all_pairs.T
            grid[rows.astype(int), cols.astype(int)] = 1

        batch.append(grid.flatten())

        if len(batch) >= batch_size:
            ipca.partial_fit(np.stack(batch))
            batch = []

    if len(batch) > 0:
        ipca.partial_fit(np.stack(batch))


def create_eigenimage_streaming(BB_directory_list, train_indices, type_indices, fold_index, type, n_components=3, batch_size=32):
    from sklearn.decomposition import IncrementalPCA
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    ipca = IncrementalPCA(n_components=n_components)

    for i in range(NUMBER_OF_SENSORS):
        files = sorted(os.listdir(BB_directory_list[i]))
        if type == 'train':
            selected_files = [files[j] for j in train_indices[i]]
        else:
            selected_files = [files[j] for j in type_indices[i]]

        process_grid_maps_streaming(selected_files, ipca, i + 1, batch_size=batch_size)

    # Get components and reshape them into (n_components, H, W)
    eigenimages = ipca.components_.reshape((n_components, Y_RANGE, X_RANGE))

    # Combine components into a single eigenimage
    # Option 1: simple sum
    combined = np.sum(eigenimages, axis=0)

    # Optional: normalize the combined image to [0, 1] for visualization
    combined_normalized = (combined - combined.min()) / (combined.max() - combined.min())

    # Plot and save the single combined eigenimage
    plt.imshow(combined_normalized, cmap='gray')
    plt.title(f"{type.capitalize()} Combined Eigenimage (Top {n_components}) - Fold {fold_index}")
    plt.axis('off')
    plt.savefig(f"{type}_combined_eigenimage_fold_{fold_index}.png")
    plt.close()

    print(f"{type.capitalize()} combined eigenimage for fold {fold_index} saved.")

    return combined_normalized



if __name__ == '__main__':
    gc.collect()
    set_start_method("spawn", force=True)
    random.seed(SEED)

    complete_name_chunck_path = os.path.join(FFCV_DIR)
    os.makedirs(complete_name_chunck_path, exist_ok=True)

    BB_directory_list = []
    for i in range(1, NUMBER_OF_SENSORS + 1):
        BB_directory_list.append(SNAPSHOT_X_GRID_DIRECTORY.replace('X', str(i)))

    for fold_index in range(K_FOLDS):
        print(f"\nStarting Fold {fold_index + 1}/{K_FOLDS}")
        train_indices_per_sensor = []
        val_indices_per_sensor = []
        test_indices_per_sensor = []

        for sensor_idx in range(NUMBER_OF_SENSORS):
            files = sorted(os.listdir(BB_directory_list[sensor_idx]))
            total_len = len(files)
            indices = np.arange(total_len)

            # Random but reproducible shuffle
            np.random.seed(SEED + fold_index)
            np.random.shuffle(indices)

            test_len = int(TEST_SIZE * total_len)
            val_len = int(VAL_SIZE * total_len)

            test_indices = indices[:test_len]
            val_indices = indices[test_len:test_len + val_len]
            train_indices = indices[test_len + val_len:]

            test_indices_per_sensor.append(test_indices)
            val_indices_per_sensor.append(val_indices)
            train_indices_per_sensor.append(train_indices)

        train_eigenimages = create_eigenimage_streaming(
            BB_directory_list, train_indices_per_sensor, None, fold_index, 'train'
        )
        val_eigenimages = create_eigenimage_streaming(
            BB_directory_list, None, val_indices_per_sensor, fold_index, 'val'
        )
        test_eigenimages = create_eigenimage_streaming(
            BB_directory_list, None, test_indices_per_sensor, fold_index, 'test'
        )