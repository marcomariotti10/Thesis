HOME = r'/mnt/c/Users/marco/Desktop/Tesi'   # <---- CHANGE THIS

DATA_HOME = HOME + r'/data_map03_0911'

HOME_DIR = HOME + r'/Codice_carla/final0911' 

LIDAR_1_DIRECTORY = DATA_HOME + r'/1_original/lidar_output_lidar1/lidar'
LIDAR_1_GRID_DIRECTORY = DATA_HOME + r'/3_lidar_grid_map/lidar1_grid'

LIDAR_2_DIRECTORY = DATA_HOME + r'/1_original/lidar_output_lidar2/lidar'
LIDAR_2_GRID_DIRECTORY = DATA_HOME + r'/3_lidar_grid_map/lidar2_grid'

LIDAR_3_DIRECTORY = DATA_HOME + r'/1_original/lidar_output_lidar3/lidar'
LIDAR_3_GRID_DIRECTORY = DATA_HOME + r'/3_lidar_grid_map/lidar3_grid'

POSITIONS_DIRECTORY = DATA_HOME + r'/1_original/position'
LIDAR_DIRECTORY = DATA_HOME + r'/1_original'

NEW_POSITION_LIDAR_1_DIRECTORY = DATA_HOME + r'/2_new_positions/new_positions_lidar1'
NEW_POSITIONS_LIDAR_1_GRID_DIRECTORY = DATA_HOME + r'/4_positions_grid_map/new_positions_lidar1_grid'
POSITION_LIDAR_1_GRID_NO_BB = DATA_HOME + r'/5_positions_grid_map_no_BB/new_positions_lidar1_grid_no_BB'

NEW_POSITION_LIDAR_2_DIRECTORY = DATA_HOME + r'/2_new_positions/new_positions_lidar2'
NEW_POSITIONS_LIDAR_2_GRID_DIRECTORY = DATA_HOME + r'/4_positions_grid_map/new_positions_lidar2_grid'
POSITION_LIDAR_2_GRID_NO_BB = DATA_HOME + r'/5_positions_grid_map_no_BB/new_positions_lidar2_grid_no_BB'

NEW_POSITION_LIDAR_3_DIRECTORY = DATA_HOME + r'/2_new_positions/new_positions_lidar3'
NEW_POSITIONS_LIDAR_3_GRID_DIRECTORY = DATA_HOME + r'/4_positions_grid_map/new_positions_lidar3_grid'
POSITION_LIDAR_3_GRID_NO_BB = DATA_HOME + r'/5_positions_grid_map_no_BB/new_positions_lidar3_grid_no_BB'

LIDAR_1_TEST = DATA_HOME + r'/6_test/lidar_test/lidar_1'
POSITION_1_TEST = DATA_HOME + r'/6_test/position_test/position_1'

LIDAR_2_TEST = DATA_HOME + r'/6_test/lidar_test/lidar_2'
POSITION_2_TEST = DATA_HOME + r'/6_test/position_test/position_2'

LIDAR_3_TEST = DATA_HOME + r'/6_test/lidar_test/lidar_3'
POSITION_3_TEST = DATA_HOME + r'/6_test/position_test/position_3'

SIMULATION_DIR = HOME_DIR + r'/simulation_data'
CONFIG_DIR = HOME_DIR + r'/config'

SCALER_DIR = HOME_DIR + r'/scalers'

MODEL_DIR = HOME_DIR + r'/models'

CHUNCKS_DIR = HOME_DIR + r'/chuncks'    

FFCV_DIR = HOME_DIR + r'/ffcv'

# For Show_grid_map
MIN_HEIGHT = -60.0

# For Conversion_BB_into_2,5D
REDUCING_RANGE = 0.5 # It can't be set too low because can be generated a bounnfing box with only two points (because we also want the bounging box of cars half out the grid)

# For Conversion_3D_to_2,5D and Conversion_BB_into_2,5D
GRID_RESOLUTION = 0.1
X_RANGE = 400   #Used also in show_grid_map
Y_RANGE = 400   #Used also in show_grid_map
X_MIN = 20      #Used also in show_grid_map
Y_MIN = 20      #Used also in show_grid_map
FLOOR_HEIGHT = -100
RANGE_FOR_ROTATED_VEHICLES = 0.1 # To identify the bounding box of rotated vehicles that are partially outside the grid

# For Show_lidar_data and Show_grid_map
LIDAR_FILE_1 = 450
LIDAR_FILE_2 = 100
LIDAR_FILE_3 = 100

# For Eliminate_BB_without_points
INCREMENT_BB_PEDESTRIAN = 2
NUM_MIN_POINTS_VEHICLE = 10
NUM_MIN_POINTS_BICYCLE = 5
NUM_MIN_POINTS_PEDESTRIAN = 1
HEIGHT_OFFSET = 3 # Height from where consider point in the bounding boxes

# For Neural_network
SEED = 42
TEST_SIZE = 0.1
NUMBER_OF_CHUNCKS = 3
NUMBER_OF_CHUNCKS_TEST = 2
