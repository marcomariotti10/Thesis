import sys
import os
from pathlib import Path

# Add scripts directory to path to import settings
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from settings import HOME

###################################### DIRECTORIES #####################################

DATA_DIR = HOME + r'/dataset'                                                  #Where dataset is saved

SENSORS_DIR = HOME + r'/code/scripts/sensors_specs'                          #Where configuration files for sensors are saved

SCALER_DIR = HOME + r'/scaler'                                                

MODEL_DIR = HOME + r'/models'                                                 

ORIGINAL_DIRECTORY = DATA_DIR + r'/Original'                                   #Where original files are saved
LIDAR_DIRECTORY = DATA_DIR + r'/Original/lidarX/lidar'
SNAPSHOT_DIRECTORY = DATA_DIR + r'/Original/positionX'

SNAPSHOT_SYNCRONIZED_DIRECTORY = DATA_DIR + r'/Syncronized_snapshot/snapshotX' #Where syncronized snapshot files are saved

HEIGHTMAP_DIRECTORY = DATA_DIR + r'/Heightmap/lidarX'                          #Where heightmap files are saved

OGM_DIRECTORY = DATA_DIR + r'/OGM/snapshotX'                                   #Where occupancy grid map files are saved

FFCV_DIRECTORY = DATA_DIR + r'/FFCV'                                           #Where ffcv files are saved