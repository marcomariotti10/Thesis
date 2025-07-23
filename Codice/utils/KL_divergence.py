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
from datetime import datetime
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import KFold
from multiprocessing import set_start_method, Process, Queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *

if platform.system() == 'Linux':
    from ffcv.loader import Loader, OrderOption
    from ffcv.fields.decoders import NDArrayDecoder
    from ffcv.transforms import ToTensor, ToDevice

K_FOLDS = 7
TEST_SIZE = 0.1
VAL_SIZE = 0.1
SEED = 42


def process_grid_maps_streaming_worker(file_list, sensor_idx, batch_size, queue):
    try:
        batch = []
        for file in tqdm(file_list, desc=f"Sensor {sensor_idx + 1}"):
            complete_path = SNAP.replace('X', str(sensor_idx + 1)) + '/' + file
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

        queue.put((sensor_idx, batch))
    except Exception as e:
        print(f"[ERROR] Sensor {sensor_idx + 1} crashed: {e}")
        queue.put((sensor_idx, []))


def create_eigenimage_streaming(BB_directory_list, train_indices_per_sensor, type_indices_per_sensor, fold_index, type, n_components=1, batch_size=8):
    ipca = IncrementalPCA(n_components=n_components)

    processes = []
    queue = Queue()

    for sensor_idx in range(NUMBER_OF_SENSORS):
        files = sorted(os.listdir(BB_directory_list[sensor_idx]))
        if type == 'train':
            selected_files = [files[j] for j in train_indices_per_sensor[sensor_idx]]
        else:
            selected_files = [files[j] for j in type_indices_per_sensor[sensor_idx]]

        print(f"[Main] Sensor {sensor_idx + 1}: selected {len(selected_files)} files for {type}")
        p = Process(target=process_grid_maps_streaming_worker, args=(selected_files, sensor_idx, batch_size, queue))
        p.start()
        processes.append(p)

    received_sensors = set()
    while len(received_sensors) < NUMBER_OF_SENSORS:
        sensor_idx, batch = queue.get()
        if sensor_idx not in received_sensors:
            print(f"[Main] Received data from Sensor {sensor_idx + 1} with {len(batch)} samples")
            for i in range(0, len(batch), batch_size):
                current_batch = batch[i:i + batch_size]
                if len(current_batch) > 0:
                    ipca.partial_fit(np.stack(current_batch))
            received_sensors.add(sensor_idx)

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            print(f"[WARNING] Sensor process {p.pid} still alive — terminating.")
            p.terminate()

    # Save eigenimages
    eigenimages = ipca.components_.reshape((n_components, Y_RANGE, X_RANGE))
    for i in range(n_components):
        plt.imshow(eigenimages[i], cmap='gray')
        plt.title(f"{type.capitalize()} Eigenimage {i + 1} Fold {fold_index}")
        plt.axis('off')
        plt.savefig(f"{type}_eigenimage_{i + 1}_fold_{fold_index}.png")
        plt.close()

    print(f"{type.capitalize()} eigenimages for fold {fold_index} saved.")
    return eigenimages


if __name__ == '__main__':
    gc.collect()
    set_start_method("spawn", force=True)
    random.seed(SEED)

    os.makedirs(FFCV_DIR, exist_ok=True)
    BB_directory_list = [.replace('X', str(i)) for i in range(1, NUMBER_OF_SENSORS + 1)]

    NUM_CHUNKS = 10
    chunk_indices_by_sensor = {}

    for sensor_idx in range(NUMBER_OF_SENSORS):
        files = sorted(os.listdir(BB_directory_list[sensor_idx]))
        indices = np.arange(len(files))
        chunks = np.array_split(indices, NUM_CHUNKS)
        chunk_indices_by_sensor[sensor_idx] = {i: chunks[i] for i in range(NUM_CHUNKS)}

        print(f"Sensor {sensor_idx + 1}: total samples = {len(files)}")
        for k, v in chunk_indices_by_sensor[sensor_idx].items():
            print(f"  Chunk {k}: [{v[0]} ... {v[-1]}] (len={len(v)})")

    for fold_index in range(K_FOLDS):
        fold_index = 4
        print(f"\n=== Starting Fold {fold_index + 1}/{K_FOLDS} ===")

        test_chunk_idx = fold_index % NUM_CHUNKS
        val_chunk_idx = (fold_index + 1) % NUM_CHUNKS
        train_chunk_indices = [i for i in range(NUM_CHUNKS) if i not in (test_chunk_idx, val_chunk_idx)]

        print(f"Fold {fold_index + 1}:")
        print(f"  Test chunk: {test_chunk_idx}")
        print(f"  Validation chunk: {val_chunk_idx}")
        print(f"  Train chunks: {train_chunk_indices}")

        train_indices_per_sensor = []
        val_indices_per_sensor = []
        test_indices_per_sensor = []

        for sensor_idx in range(NUMBER_OF_SENSORS):
            chunks = chunk_indices_by_sensor[sensor_idx]

            test_indices = chunks[test_chunk_idx]
            val_indices = chunks[val_chunk_idx]
            train_indices = np.concatenate([chunks[i] for i in train_chunk_indices])

            train_indices_per_sensor.append(train_indices)
            val_indices_per_sensor.append(val_indices)
            test_indices_per_sensor.append(test_indices)

            print(f"\nSensor {sensor_idx + 1}:")
            print(f"  Test indices:      [{test_indices[0]} ... {test_indices[-1]}]")
            print(f"  Validation indices:[{val_indices[0]} ... {val_indices[-1]}]")
            print(f"  Train indices:     [{train_indices[0]} ... {train_indices[-1]}]")

        train_eigenimages = create_eigenimage_streaming(BB_directory_list, train_indices_per_sensor, None, fold_index, 'train')
        val_eigenimages = create_eigenimage_streaming(BB_directory_list, None, val_indices_per_sensor, fold_index, 'val')
        test_eigenimages = create_eigenimage_streaming(BB_directory_list, None, test_indices_per_sensor, fold_index, 'test')





