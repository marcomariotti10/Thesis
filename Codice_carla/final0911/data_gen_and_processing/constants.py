HOME = r'C:\Users\marco\Desktop\Tesi'   # <---- CHANGE THIS

DATA_HOME = HOME + r'\data_map03_0911'

HOME_DIR = HOME + r'\Codice_carla\final - Copy' 

LIDAR_1_DIRECTORY = DATA_HOME + r'\1_original\lidar_output_lidar1\lidar'
LIDAR_1_GRID_DIRECTORY = DATA_HOME + r'\3_lidar_grid_map\lidar1_grid'

LIDAR_2_DIRECTORY = DATA_HOME + r'\1_original\lidar_output_lidar2\lidar'
LIDAR_2_GRID_DIRECTORY = DATA_HOME + r'\3_lidar_grid_map\lidar2_grid'

LIDAR_3_DIRECTORY = DATA_HOME + r'\1_original\lidar_output_lidar3\lidar'
LIDAR_3_GRID_DIRECTORY = DATA_HOME + r'\3_lidar_grid_map\lidar3_grid'

POSITIONS_DIRECTORY = DATA_HOME + r'\1_original\position'
LIDAR_DIRECTORY = DATA_HOME + r'\1_original'

NEW_POSITION_LIDAR_1_DIRECTORY = DATA_HOME + r'\2_new_positions\new_positions_lidar1'
NEW_POSITIONS_LIDAR_1_GRID_DIRECTORY = DATA_HOME + r'\4_positions_grid_map\new_positions_lidar1_grid'
POSITION_LIDAR_1_GRID_NO_BB = DATA_HOME + r'\5_positions_grid_map_no_BB\new_positions_lidar1_grid_no_BB'

NEW_POSITION_LIDAR_2_DIRECTORY = DATA_HOME + r'\2_new_positions\new_positions_lidar2'
NEW_POSITIONS_LIDAR_2_GRID_DIRECTORY = DATA_HOME + r'\4_positions_grid_map\new_positions_lidar2_grid'
POSITION_LIDAR_2_GRID_NO_BB = DATA_HOME + r'\5_positions_grid_map_no_BB\new_positions_lidar2_grid_no_BB'

NEW_POSITION_LIDAR_3_DIRECTORY = DATA_HOME + r'\2_new_positions\new_positions_lidar3'
NEW_POSITIONS_LIDAR_3_GRID_DIRECTORY = DATA_HOME + r'\4_positions_grid_map\new_positions_lidar3_grid'
POSITION_LIDAR_3_GRID_NO_BB = DATA_HOME + r'\5_positions_grid_map_no_BB\new_positions_lidar3_grid_no_BB'

SIMULATION_DIR = HOME_DIR + r'\simulation_data'
CONFIG_DIR = HOME_DIR + r'\config'


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

# For Show_lidar_data and Show_grid_map
LIDAR_FILE_1 = "20250126_112202_999725_004102"
LIDAR_FILE_2 = "20250118_102636_727556_168423"
LIDAR_FILE_3 = "20250118_102716_454293_169283"

# For Eliminate_BB_without_points
INCREMENT_BB_PEDESTRIAN = 2
NUM_MIN_POINTS_VEHICLE = 10
NUM_MIN_POINTS_PEDESTRIAN = 1
HEIGHT_OFFSET = 3 # Height from where consider point in the bounding boxes

# For Neural_network
SEED = 42
TEST_SIZE = 0.1
