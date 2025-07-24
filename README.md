# Tesi - Neural Network Pipeline for Dataset Generation, Preprocessing, and Training

This project provides a complete pipeline for generating synthetic LiDAR data, preprocessing it into occupancy grids, splitting and preparing it into datasets, training neural networks (including autoencoders), and visualizing the results.

## 📁 Project Structure

### `config/`
Configuration scripts and sensor map parameters.
- `constants.py`: Stores global constants used across modules.
- `functions.py`: Helper functions for general configuration use.
- `map_03_final/`: Contains `.json` files describing LiDAR sensor layouts for a specific map.
- `__pycache__/`: Auto-generated Python cache.

---

### `data_geneneration/`
Modules for generating synthetic data from simulation environments.
- `generate_simulation.py`: Main script to launch a full simulation and capture data.
- `local_sim_utility.py`: Local utility functions used during simulation generation.
- `save_snapshot.py`: Handles saving of environment snapshots.
- `sensor.py`: Defines LiDAR sensor parameters and data structure.
- `spawn_objects.py`: Adds dynamic objects (cars, pedestrians, etc.) into the simulated world.
- `utility.py`: Generic utilities for data generation.
- `__pycache__/`: Compiled Python files.

---

### `preprocessing/`
Handles transformation of raw LiDAR data into usable formats for neural networks.
- `filter_grid.py`: Applies filters to clean grid maps.
- `LiDAR_grid.py`: Converts LiDAR scans to 2D grid representations.
- `occupancy_grid.py`: Generates occupancy grid maps from LiDAR data.
- `syncronization.py`: Aligns sensor readings over time.
- `start.py`: Entry point to run the full preprocessing pipeline.
- `__init__.py`: Marks this folder as a Python module.

---

### `dataset_division/`
Takes preprocessed data and prepares it for training.
- `generate_chuncks.py`: Splits datasets into chunks for k-fold cross-validation.
- `generate_ffcv.py`: Converts data into `.ffcv` format for fast loading.
- `augmentation.py`: Applies augmentations to training data.
- `combine_augmentation.py`: Combines multiple augmentation strategies.
- `scaler.py`: Normalizes or standardizes data.
- `start.py`: Executes dataset splitting and export pipeline.
- `__init__.py`: Module file.

---

### `neural_network/`
Definition and training of the neural network models.
- `NN_autoencoder.py`: Defines a single-channel autoencoder network.
- `NN_autoencoder_multi.py`: Multi-channel variant of the autoencoder.
- `train_encoder.py`: Main script to train the neural network models.
- `ensemble.py`: Logic for combining predictions from multiple models (ensembles).
- `__pycache__/`: Cached compiled Python files.

---

### `utils/`
Support scripts for data analysis and training utility.
- `eigenimage.py`: Computes principal components (eigenimages) from the data.
- `compare_eigenimage.py`: Compares eigenimages for different sensors or datasets.
- `grid_search.py`: Utility for performing grid search over hyperparameters.
- `__init__.py`: Python module marker.

---

### `visualization/`
Visual output generation from data and models.
- `results_autoencoder.py`: Plots loss curves, reconstructions, and metrics.
- `show_grid_map.py`: Displays grid maps (2D projections).
- `show_lidar_data.py`: Visualization of raw LiDAR point clouds.
- `visualize_predictions.py`: Shows predictions vs. ground truth from trained models.
- `__init__.py`: Module file.
- `__pycache__/`: Compiled caches.

---

## 🚀 Workflow Overview

1. **Data Generation**:  
   Run `data_geneneration/generate_simulation.py` to generate synthetic LiDAR data with sensor configurations from `config/map_03_final/`.

2. **Preprocessing**:  
   Use `preprocessing/start.py` to transform the raw data into occupancy grid maps.

3. **Dataset Preparation**:  
   Use `dataset_division/start.py` to split, augment, and convert the data into train/val/test sets in FFCV format.

4. **Training**:  
   Launch model training via `neural_network/train_encoder.py`.

5. **Visualization and Analysis**:  
   Use scripts in `visualization/` and `utils/` to visualize results, analyze grid maps, and explore eigenimages.

---

## 📌 Notes
- Python 3.7+ is recommended.
- Ensure simulation tools or CARLA (if used) are properly installed for data generation.
