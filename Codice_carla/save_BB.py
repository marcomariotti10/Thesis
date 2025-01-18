import carla
import random
import json
import time

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
    
    all_actors = world.get_actors()
    ped = world.get_actors().filter('walker.*')
    veh = world.get_actors().filter('vehicle.*')
    sen = world.get_actors().filter('sensor.*')

    pedestrians = []
    vehicles = []
    sensors = []

    for p in ped :
        pedestrians.append(p.id)

    for v in veh :
        vehicles.append(v.id)

    for s in sen :
        sensors.append(s.id)
    

    # Retrieve positions of all actors
    id = []
    actor_positions_1 = []
    actor_positions_2 = []
    actor_positions_3 = []
    actor_positions_4 = []
    actor_positions_5 = []
    actor_positions_6 = []
    actor_positions_7 = []
    actor_positions_8 = []
    actor_center = []
    labels = []

    for actor in all_actors:
        location = actor.bounding_box.get_world_vertices(actor.get_transform())
        local_center = actor.bounding_box.location
        center = actor.get_transform().transform(local_center)
        actor_positions_1.append(location[0])
        actor_positions_2.append(location[1])
        actor_positions_3.append(location[2])
        actor_positions_4.append(location[3])
        actor_positions_5.append(location[4])
        actor_positions_6.append(location[5])
        actor_positions_7.append(location[6])
        actor_positions_8.append(location[7])
        actor_center.append(center)
        id.append(actor.id)
    
    for ids in id:
        if (ids in pedestrians) : labels.append("pedestrian")
        elif (ids in vehicles) : labels.append("vehicle")
        elif (ids in sensors) : labels.append("sensor")
        else : labels.append("other")
    
    filename=('C:/Users/airlab/Desktop/CARLA/multi_sensor/carla-sim-static-sensors/data/position/%s.csv' % (datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    # Prepare the data in a dictionary format
    data = [{"actor_id": actor_id, "label" : label,"actor_center_x" : actor_cent.x,"actor_center_y" : actor_cent.y, "actor_center_z" : actor_cent.z,  "actor_location_1x": actor_location_1.x,  "actor_location_1y": actor_location_1.y, "actor_location_1z": actor_location_1.z, "actor_location_2x": actor_location_2.x, "actor_location_2y": actor_location_2.y,"actor_location_2z": actor_location_2.z, "actor_location_3x": actor_location_3.x, "actor_location_3y": actor_location_3.y, "actor_location_3z": actor_location_3.z, "actor_location_4x": actor_location_4.x,  "actor_location_4y": actor_location_4.y, "actor_location_4z": actor_location_4.z, "actor_location_5x": actor_location_5.x,  "actor_location_5y": actor_location_5.y, "actor_location_5z": actor_location_5.z, "actor_location_6x": actor_location_6.x,  "actor_location_6y": actor_location_6.y, "actor_location_6z": actor_location_6.z, "actor_location_7x": actor_location_7.x,  "actor_location_7y": actor_location_7.y, "actor_location_7z": actor_location_7.z, "actor_location_8x": actor_location_8.x,  "actor_location_8y": actor_location_8.y, "actor_location_8z": actor_location_8.z, } for actor_id, label, actor_cent, actor_location_1,  actor_location_2, actor_location_3, actor_location_4, actor_location_5, actor_location_6, actor_location_7, actor_location_8 in zip(id, labels, actor_center, actor_positions_1, actor_positions_2,  actor_positions_3,  actor_positions_4,  actor_positions_5,  actor_positions_6,  actor_positions_7,  actor_positions_8) if (label !="other" and label !='sensor')]

    with open(filename, 'w', newline='') as file:
        fieldnames = ['actor_id', 'label', 'actor_center_x','actor_center_y', 'actor_center_z', 'actor_location_1x', 'actor_location_1y', 'actor_location_1z', 'actor_location_2x', 'actor_location_2y', 'actor_location_2z', 'actor_location_3x', 'actor_location_3y', 'actor_location_3z', 'actor_location_4x', 'actor_location_4y', 'actor_location_4z', 'actor_location_5x', 'actor_location_5y', 'actor_location_5z', 'actor_location_6x', 'actor_location_6y', 'actor_location_6z', 'actor_location_7x', 'actor_location_7y', 'actor_location_7z', 'actor_location_8x', 'actor_location_8y', 'actor_location_8z', ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Write the header 
        writer.writeheader() 
        # Write the rows 
        writer.writerows(data)

if __name__ == "__main__":

    f = open(CONFIG_DIR + '/sensors.json')
    sensors_config = json.load(f)
    random.seed(SEED)

    print("Connecting to server...")

    client = carla.Client(IP, PORT)
    client.set_timeout(150.0)
    world = client.get_world()

    try:

        while True:
            retrieve_actor_positions()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Execution interrupted by user (Ctrl + C).")

    except Exception as e: 
        print(f"An error occurred: {e}")

    finally:
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting.")


