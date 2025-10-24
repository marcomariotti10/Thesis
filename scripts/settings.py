import os
import platform
from pathlib import Path

# Root directory path for the thesis project - CHANGE THIS TO YOUR PROJECT DIRECTORY
HOME = r'/mnt/c/Users/marco/Desktop/Thesis'

####################################### SIMULATION #####################################

# Random seed for reproducible simulation results
SEED = 42

# Port number for the CARLA simulation server connection
SERVER_PORT = 2000

# Port number for the CARLA traffic manager
TRAFFIC_MANAGER_PORT = 8000

# IP address of the simulation server (localhost)
IP = "127.0.0.1"

# Maximum number of vehicles to spawn in the simulation environment
NUMBER_VEHICLES = 40

# Maximum number of pedestrians to spawn in the simulation environment
NUMBER_PEDESTRIANS = 40

####################################### PREPROCESSING #####################################

# Total number of LiDAR sensors used in the simulation setup
NUMBER_OF_SENSORS = 12

# Resolution of the occupancy grid in meters per cell
GRID_RESOLUTION = 0.1

# X-dimension of the occupancy grid in number of cells
DIMENSION_X = 400

# Y-dimension of the occupancy grid in number of cells  
DIMENSION_Y = 400

# Value assigned to cells with no LiDAR data/measurements
NO_DATA_VALUE = -100

# Additional margin added to pedestrian bounding boxes for better detection
INCREMENT_BB_PEDESTIAN = 0.12

# Minimum number of LiDAR points required to classify an object as a vehicle
MINIMUM_POINTS_VEHICLE = 5

# Minimum number of LiDAR points required to classify an object as a bicycle
MINIMUM_POINTS_BICYCLE = 3

# Minimum number of LiDAR points required to classify an object as a pedestrian
MINIMUM_POINTS_PEDESTRIAN = 1

# Percentage of the dataset to be augmented with transformations
AUGMENTATION_PERCENTAGE = 0.3

# Probability that augmented samples will undergo rotation transformation
ROTATION_PROBABILITY = 0.5

# Maximum rotation angle applied during data augmentation (in degrees)
MAX_ROTATION_ANGLE = 45

# Minimum rotation angle applied during data augmentation (in degrees)
MINIMUM_ROTATION_ANGLE = 30

# Probability that augmented samples will undergo spatial shift transformation
SHIFT_PROBABILITY = 0.2

# Maximum spatial shift distance during augmentation (in grid cells)
MAX_SHIFT = 100

# Minimum spatial shift distance during augmentation (in grid cells)
MINIMUM_SHIFT = 50

# Probability that augmented samples will undergo horizontal/vertical flipping
FLIP_PROBABILITY = 0.3

# Minimum number of detections of an actor to be included into the target samples
MINIMUM_NUMBER_OF_DETECTIONS = 1

# Percentage of data reserved for testing (validation split)
TEST_SIZE_PERCENTAGE = 0.1

# Number of data chunks to create for training dataset
NUMBER_OF_CHUNCKS_TRAIN = 2

# Number of data chunks to create for testing and validating dataset
NUMBER_OF_CHUNCKS_TEST = 1

# Number of consecutive frames used as input to the model
NUMBER_FRAMES_INPUT = 5

# Future time steps to predict relative to the last input frame
FUTURE_TARGET_RILEVATION = [5,10]

####################################### MODEL TRAINING #####################################

# Number of samples processed together in each training batch
BATCH_SIZE = 4

# Learning rate for the optimizer during model training
LEARNING_RATE = 0.001

# Total number of training epochs to run
NUMBER_EPOCHS = 100

# Number of epochs to wait before early stopping if no improvement is observed
PATIENCE_EARLY_STOPPING = 5

# Number of epochs to wait before reducing learning rate if no improvement
PATIENCE_LR_SCHEDULER = 3

# Minimum value for the noise schedule beta parameter in diffusion models
MINIMUM_BETHA = 1e-4

# Maximum value for the noise schedule beta parameter in diffusion models
MAXIMUM_BETHA = 0.02

# Total number of diffusion timesteps in the forward noising process
RANGE_TIMESTEPS = 1000