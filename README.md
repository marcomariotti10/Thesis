# Vehicle/people detection and tracking for V2I cooperation and smart traffic management

This project provides a complete pipeline for generating synthetic LiDAR data via CARLA simulation, preprocessing it into heightmaps, and training a segmentation multi-head neural network for prediction of the future positions of actors at multiple time instant. Also, visualization of the predictions and model performance.

## 📁 Project Structure

### `config/`
Configuration scripts and sensor map parameters.
- `constants.py`: Stores global constants and path used across modules.
- `functions.py`: General functions used across multiple modules.
- `map_03_final/`: Contains `.json` files describing LiDAR sensor layouts for a CARLA map '10 HD'.

---

### `data_geneneration/`
Modules for generating synthetic data from simulation environments.
- `generate_simulation.py`: Main script to load the world inside CARLA simulation.
- `local_sim_utility.py`: Local utility functions used during simulation generation.
- `save_snapshot.py`: Handles the saving of simulations snapshots to capture actors bounding boxes.
- `sensor.py`: Spawn lidar sensors inside the simulation.
- `spawn_objects.py`: Adds dynamic objects (cars, pedestrians, etc.) into the simulated world.
- `utility.py`: Generic utilities for data generation.

---

### `preprocessing/`
Handles transformation of raw LiDAR data into usable formats for neural networks.
- `filter_grid.py`: Applies filtering ro remove bounfing boxes without sufficient lidar point inside.
- `LiDAR_grid.py`: Converts LiDAR scans to heightmap representations.
- `occupancy_grid.py`: Generates occupancy grid maps from simulation snapshot.
- `syncronization.py`: Aligns sensor readings with simulation snapshots over time.
- `start.py`: Entry point to run the full preprocessing pipeline.

---

### `dataset_division/`
Takes preprocessed data and prepares it for training.
- `generate_chuncks.py`: Splits datasets into chunks.
- `generate_ffcv.py`: Converts data into `.ffcv` format for fast loading.
- `augmentation.py`: Applies augmentations to training data.
- `combine_augmentation.py`: Combines standard data with augmented data.
- `scaler.py`: Normalization of the data.
- `start.py`: Executes dataset splitting and export pipeline.

---

### `neural_network/`
Definition and training of the neural network models.
- `NN_autoencoder.py`: Training og a single-head autoencoder network for the prediction of one single future instant.
- `NN_autoencoder_multi.py`: Multi-channel variant of the autoencoder for multiple simultaneous predictions.
- `train_encoder.py`: Main script to train the neural network encoder, without the heads.
- `ensemble.py`: Logic for combining predictions from multiple models (ensembles).

---

### `utils/`
Support scripts for data analysis and training utility.
- `eigenimage.py`: Computes principal components (eigenimages) from the train, test and validation set.
- `compare_eigenimage.py`: Compares eigenimages and calculate distance metrics.
- `grid_search.py`: Grid search over hyperparameters of the network.

---

### `visualization/`
Visual output generation from data and models.
- `results_autoencoder.py`: Plots confusion matrix and metrics.
- `show_grid_map.py`: Visualization of heightmaps and bounding boxes.
- `show_lidar_data.py`: Visualization of raw LiDAR point clouds and bounding boxes.
- `visualize_predictions.py`: Shows predictions vs. ground truth from trained models.

---

## 🚀 Workflow Overview

1. **Data Generation**:  
   Run `data_geneneration/generate_simulation.py` to generate synthetic LiDAR data.

2. **Preprocessing**:  
   Use `preprocessing/start.py` to transform the raw data into heightmap(inputs) and occupancy grid maps (targets).

3. **Dataset Preparation**:  
   Use `dataset_division/start.py` to split, augment, and convert the data into train/val/test sets in FFCV format.

4. **Training**:  
   Launch model training via `NN_autoencoder.py` or `NN_autoencoder_multi.py`.

5. **Visualization and Analysis**:  
   Use scripts in `visualization/` and to visualize results.

---

## 📌 Notes
- Python 3.7+ is recommended.
- Ensure simulation tools or CARLA (if used) are properly installed for data generation.
