import carla
import random
import json
import time
import os

from constants import *
from utility import *
from local_sim_utility import *
from spawn_objects import *

from datetime import datetime
import csv

LOCAL = True
SEED = 42
PORT = 2000
IP = "127.0.0.1"
    

def retrieve_actor_positions():
    # Get all actors in the simulation
    
    pedestrians = []
    car = []
    sensors = []
    bicycles = []
    trucks = []
    vans = []
    motorcycles = []

    blueprints = []

    all_actors = world.get_actors()
    blueprints = [bp for bp in world.get_blueprint_library().filter('*')]
    for blueprint in blueprints:
        print(blueprint.id)
    for attr in blueprint:
        print('  - {}'.format(attr))

    
    for i in range (len(blueprints)):
        if (blueprints[i].get_attribute('base_type') == 'car'):
            car.append(all_actors[i].id)
        elif (blueprints[i].get_attribute('base_type') == 'walker'):
            pedestrians.append(all_actors[i].id)
        elif (blueprints[i].get_attribute('base_type') == 'sensor'):
            sensors.append(all_actors[i].id)
        elif (blueprints[i].get_attribute('base_type') == 'bicycle'):
            bicycles.append(all_actors[i].id)
        elif (blueprints[i].get_attribute('base_type') == 'truck'):
            trucks.append(all_actors[i].id)
        elif (blueprints[i].get_attribute('base_type') == 'van'):
            vans.append(all_actors[i].id)
        elif (blueprints[i].get_attribute('base_type') == 'motorcycle'):
            motorcycles.append(all_actors[i].id)
    

    # Retrieve positions of all actors
    id = []
    labels = []
    actor_center = []
    dimensions = []
    rotations = []

    for actor in all_actors:
        
        local_center = actor.bounding_box.location
        center = actor.get_transform().transform(local_center)
        dimension = actor.bounding_box.extent

        rotation = actor.get_transform().rotation
        
        rotations.append(rotation)
        
        dimensions.append(dimension)
        actor_center.append(center)
        id.append(actor.id)
    
    for ids in id:
        if (ids in pedestrians) : labels.append("pedestrian")
        elif (ids in car) : labels.append("car")
        elif (ids in sensors) : labels.append("sensor")
        elif (ids in bicycles) : labels.append("bicycle")
        elif (ids in trucks) : labels.append("truck")
        elif (ids in vans) : labels.append("van")
        elif (ids in motorcycles) : labels.append("motorcycle")
        else : labels.append("other")

    # Create the new folder if it doesn't exist
    if not os.path.exists(POSITIONS_DIRECTORY):
        os.makedirs(POSITIONS_DIRECTORY)
    
    filename=(POSITIONS_DIRECTORY + '/%s.csv' % (datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    # Prepare the data in a dictionary format
    data = [{"actor_id": actor_id, "label" : label,"actor_center_x" : actor_cent.x,"actor_center_y" : actor_cent.y, "actor_center_z" : actor_cent.z,  "actor_dimension_x": dim.x,  "actor_dimension_y": dim.y, "actor_dimension_z": dim.z, "actor_rotation_pitch" : rot.pitch, "actor_rotation_roll" : rot.roll, "actor_rotation_yaw" : rot.yaw } for actor_id, label, actor_cent, dim, rot in zip(id, labels, actor_center, dimensions, rotations) if (label !='other' and label !='sensor')]

    with open(filename, 'w', newline='') as file:
        fieldnames = ['actor_id', 'label', 'actor_center_x','actor_center_y', 'actor_center_z', 'actor_dimension_x', 'actor_dimension_y', 'actor_dimension_z', 'actor_rotation_pitch', 'actor_rotation_roll', 'actor_rotation_yaw']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Write the header 
        writer.writeheader() 
        # Write the rows 
        writer.writerows(data)

if __name__ == "__main__":

    print("Connecting to server...")

    client = carla.Client(IP, PORT)
    client.set_timeout(150.0)
    world = client.get_world()

    print("Press Ctrl + C to stop the execution.")

    try:

        while True:
            retrieve_actor_positions()
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Execution interrupted by user (Ctrl + C).")

    except Exception as e: 
        print(f"An error occurred: {e}")

    finally:
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting.")