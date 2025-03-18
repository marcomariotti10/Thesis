import os
import platform
def get_home_directory():
    if platform.system() in 'Linux':
        return r'/mnt/c/Users/marco/Desktop/Tesi'
    else:
        return r'C:/Users/marco/Desktop/Tesi'

HOME = get_home_directory()   # <---- CHANGE THIS

DATA_HOME = HOME + r'/data_map03_0911'

HOME_SAVING = HOME + r'/Savings'

HOME_DIR = HOME + r'/Codice'

NUMBER_OF_SENSORS = 3

LIDAR_X_DIRECTORY = DATA_HOME + r'/1_original/lidar_output_lidarX/lidar'
LIDAR_X_GRID_DIRECTORY = DATA_HOME + r'/3_lidar_grid_map/lidarX_grid'

POSITIONS_DIRECTORY = DATA_HOME + r'/1_original/position'
LIDAR_DIRECTORY = DATA_HOME + r'/1_original'

NEW_POSITION_LIDAR_X_DIRECTORY = DATA_HOME + r'/2_new_positions/new_positions_lidarX'
NEW_POSITIONS_LIDAR_X_GRID_DIRECTORY = DATA_HOME + r'/4_positions_grid_map/new_positions_lidarX_grid'
POSITION_LIDAR_X_GRID_NO_BB = DATA_HOME + r'/5_positions_grid_map_no_BB/new_positions_lidarX_grid_no_BB'

LIDAR_X_TEST = DATA_HOME + r'/6_test/lidar_test/lidar_X'
POSITION_X_TEST = DATA_HOME + r'/6_test/position_test/position_X'

LIDAR_X_VAL = DATA_HOME + r'/7_val/lidar_val/lidar_X'
POSITION_X_VAL = DATA_HOME + r'/7_val/position_val/position_X'

SIMULATION_DIR = HOME_DIR + r'/simulation_data'
CONFIG_DIR = HOME_DIR + r'/config'

CHUNCKS_DIR = DATA_HOME + r'/chuncks_pred'    

FFCV_DIR = DATA_HOME + r'/ffcv_pred'

SCALER_DIR = HOME_SAVING + r'/scalers'

MODEL_DIR = HOME_SAVING + r'/models_pred'

# For save_positions
NEW_POSITIONS_OFFSETS = [(70.15, 11.50, -6.0), (100.59, -27.46, -6.0), (96.83, 6.31, -6.0), (54.05, -41.25, -6.0), (62.50, -14.60, -6.0), (36.35, -7.55, -6.0)]

# For Show_grid_map
MIN_HEIGHT = -60.0

# For Conversion_BB_into_2,5D
REDUCING_RANGE = 0.5 # It can't be set too low because can be generated a bounnfing box with only two points (because we also want the bounging box of cars half out the grid)
INCREASE_GRID_RANGE = 200

# For Conversion_3D_to_2,5D and Conversion_BB_into_2,5D
GRID_RESOLUTION = 0.1
X_RANGE = 400   #Used also in show_grid_map
Y_RANGE = 400   #Used also in show_grid_map
X_MIN = 20      #Used also in show_grid_map
Y_MIN = 20      #Used also in show_grid_map
FLOOR_HEIGHT = -100
RANGE_FOR_ROTATED_VEHICLES = 0.1 # To identify the bounding box of rotated vehicles that are partially outside the grid

# For Show_lidar_data and Show_grid_map
LIDAR_FILE_X = [370, 100, 562]

# For Eliminate_BB_without_points
INCREMENT_BB_PEDESTRIAN = 2
NUM_MIN_POINTS_VEHICLE = 5
NUM_MIN_POINTS_BICYCLE = 3
NUM_MIN_POINTS_PEDESTRIAN = 1
HEIGHT_OFFSET = 3 # Height from where consider point in the bounding boxes

# For Neural_network
SEED = 42
TEST_SIZE = 0.1
NUMBER_OF_CHUNCKS = 3
NUMBER_OF_CHUNCKS_TEST = 2
