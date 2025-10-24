import glob
import os
import sys
try:
    sys.path.append(glob.glob('C:/Users/marco/Desktop/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from data_generation.spawn_actors import *
import time
import carla
import random
import json
import argparse

LOCAL = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate a sensor in the simulation.')
    parser.add_argument('--number', type=int, default=1, help='Number of the sensor to generate')
    args = parser.parse_args()

    sensor_number = args.number
    
    # Construct the correct path to the sensor specification file
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # Go up to project root
    sensor_specs_dir = os.path.join(script_dir, 'scripts', 'sensors_specs')
    sensor_file_path = os.path.join(sensor_specs_dir, f'lidar{sensor_number}.json')
    
    with open(sensor_file_path, 'r') as f:
        sensors_config = json.load(f)
    random.seed(SEED)

    print("Connecting to server...")

    client = carla.Client(IP, SERVER_PORT)
    client.set_timeout(150.0)
    world = client.get_world()

    try:

        tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)

        # ---------------------
        # STATIC SENSORS
        # ---------------------
        print("Static sensors creation...")
        threads = []
        sensors = []
        static = []
        spawn_static_sensors(sensors_config, sensors, static, world, LOCAL)
 
        
        # -------------------
        # SIMULATION
        # -------------------
        print("Setting up the simulation...")

        print("\n\nStart the simulation!\n")
        print("Press Ctrl + C to stop the execution.")
        while True:
            if LOCAL:
                world.tick()
            else:
                world.wait_for_tick()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user (Ctrl+C)")
    except Exception as e: 
        print(e)
        
        # Destroy sensors
        for sensor in sensors:
            sensor.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('sensor.lidar.ray_cast')])
        # Destroy sensors home actors
        for st in static:
            st.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*static*')])