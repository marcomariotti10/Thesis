import tensorflow as tf
import csv
import os
import numpy as np
from tensorflow import keras as tfk
from keras import layers as tfkl
import sys
from data_creation import *

# Import constants and catch any import errors
from link_to_constants import *
link_constants()
try:
    from constants import * # type: ignore
    print("Successfully imported constants.")
except ImportError as e:
    print(f"Error importing constants: {e}")

if __name__ == "__main__":

    complete_grid_maps = []
    complete_grid_maps_BB = []

    # Load sensor1
    grid_maps = generate_grid_map(LIDAR_1_GRID_DIRECTORY) # type: ignore
    grid_maps_BB = generate_grid_map_BB(NEW_POSITION_LIDAR_1_DIRECTORY) # type: ignore

    grid_maps.append(complete_grid_maps)
    grid_maps_BB.append(complete_grid_maps_BB)
    
    # Load sensor2
    grid_maps = generate_grid_map(LIDAR_2_GRID_DIRECTORY) # type: ignore
    grid_maps_BB = generate_grid_map_BB(NEW_POSITION_LIDAR_2_DIRECTORY) # type: ignore

    grid_maps.append(complete_grid_maps)
    grid_maps_BB.append(complete_grid_maps_BB)

    # Load sensor3
    grid_maps = generate_grid_map(LIDAR_3_GRID_DIRECTORY) # type: ignore
    grid_maps_BB = generate_grid_map_BB(NEW_POSITION_LIDAR_3_DIRECTORY) # type: ignore

    grid_maps.append(complete_grid_maps)
    grid_maps_BB.append(complete_grid_maps_BB)

    # Split the data
    X_train_val, X_test, y_train_val, y_test = split_data(complete_grid_maps, complete_grid_maps_BB)
    X_train, X_val, y_train, y_val = split_data(complete_grid_maps, complete_grid_maps_BB, test_size = len(X_test)) # Esure that val and test set have the same lenght
    
    shape_input = (X_RANGE,Y_RANGE,1) # type: ignore
    
    # Input layer for the 2.5D grid map
    grid_map_input = tfkl.Input(shape_input, name='grid_map_input')

    # Encoder
    x = tfkl.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_map_input)
    x = tfkl.MaxPooling2D((2, 2), padding='same')(x)
    x = tfkl.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tfkl.MaxPooling2D((2, 2), padding='same')(x)
    x = tfkl.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = tfkl.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = tfkl.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = tfkl.UpSampling2D((2, 2))(x)
    x = tfkl.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tfkl.UpSampling2D((2, 2))(x)
    x = tfkl.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tfkl.UpSampling2D((2, 2))(x)
    decoded = tfkl.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Create the model
    model = tfkl.Model(inputs=grid_map_input, outputs=decoded)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Print the model summary
    model.summary()

    #TODO
    # Normalize the height values
    # Augment data
