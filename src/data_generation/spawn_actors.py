import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import carla
import random

from datetime import datetime

# ---------------------
# STATIC SENSORS
# ---------------------

def spawn_static_sensors(sensors_config, sensor_list, static_list, world : carla.World, LOCAL=False):
    bp_lib = world.get_blueprint_library()
    for sensors_home in sensors_config["actors"]:
        bp = bp_lib.find('static.prop.colacan')
        sensors_home_trans = carla.Transform(carla.Location(**sensors_home["location"]), carla.Rotation(**sensors_home["rotation"]))
        bp.set_attribute('role_name',    sensors_home["name"])
        sensors_home_actor = world.spawn_actor(bp, sensors_home_trans)
        for sensor in sensors_home["sensors"]:
            params = sensor["bp_parameters"]
            complete_name = sensors_home["name"] + "/" + params["role_name"]
            print(f"Creating {complete_name}...")

            sensor_trans = carla.Transform(carla.Location(**sensor["location"]), carla.Rotation(**sensor["rotation"]))
            bp = bp_lib.find(sensor["bp_type"])
            if(sensor["bp_type"].startswith("sensor.camera")):
                params = convert_to_usable_resolution(params)
            for key, value in params.items():
                bp.set_attribute(key, value)
            sensor_actor = world.spawn_actor(bp, sensor_trans, attach_to=sensors_home_actor)
            if LOCAL:
                s_type = sensor["bp_type"]

                if(s_type == "sensor.lidar.ray_cast"):

                    sensor_actor.listen(lambda point_cloud: point_cloud.save_to_disk(ORIGINAL_DIRECTORY + f'/{complete_name}' + '/%s.ply' % (datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
                    pass
            sensor_list.append(sensor_actor)
            static_list.append(sensors_home_actor)

# ---------------------
# PEDESTRIANS
# ---------------------
            
def get_walker_spawn_location(world):
        walker_area_limits = {
            'tl': {
                'x': -49.0,
                'y': +55.0,
                },
            'br': {
                'x': +59.0,
                'y': -128.0,
                },
        }
        kiosk_location = {
            'tl': {
                'x': 25.0,
                'y': -14.0,
                },
            'br': {
                'x': +35.0,
                'y': -22.0,
                },
        }
        location = world.get_random_location_from_navigation()
        return location

def spawn_pedestrians(client : carla.Client, num_ped=100, LOCAL=False):
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    # Get the blueprint for pedestrians
    pedestrians_bp = bp_lib.filter('walker.pedestrian.*')
    controller_bp = bp_lib.find('controller.ai.walker')

    # Take all the random locations to spawn
    spawn_points = []
    for i in range(num_ped):
        spawn_point = carla.Transform()
        spawn_point.location = get_walker_spawn_location(world)
        spawn_points.append(spawn_point)

    # Spawn the pedestrians
    batch = [carla.command.SpawnActor(random.choice(pedestrians_bp), spawn_point) for spawn_point in spawn_points]
    results = client.apply_batch_sync(batch, True)

    # Spawn walker AI controllers for each walker if there is no error in spawning
    batch = [carla.command.SpawnActor(controller_bp, carla.Transform(), result.actor_id) for result in results if not result.error]
    results = client.apply_batch_sync(batch, True)

    controllers_id = [result.actor_id for result in results if not result.error]
    controllers = world.get_actors(controllers_id)

    if LOCAL:
        world.tick()
    else:
        # Wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.wait_for_tick()

    ctr_list = []
    # Initialize each controller and set target to walk to
    for ctr in controllers:
        ctr.start()
        ctr.go_to_location(get_walker_spawn_location(world))
        ctr.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
        ctr_list.append(ctr) # Do this to avoid returnin an actor list object, return a list instead

    return ctr_list

# ---------------------
# VEHICLES
# ---------------------

def spawn_vehicles(client : carla.Client, tm, num_vehicles=NUMBER_VEHICLES, LOCAL=False):
    world=client.get_world()
    bp_lib = world.get_blueprint_library()

    # Define the limits of the internal road in coordinates 
    br = {
        'x': 0.0,
        'y': 0.0
    }
    tl = {
        'x': 0.1,
        'y': 0.1
    }
    vehicles_bp = bp_lib.filter('vehicle')
    # Remove Trucks, Van and Bus
    vehicles_to_remove = ['', 'motorcycle', 'bicycle', 'truck', 'van', 'bus', 'Bus', 'Truck', 'Van', 'Motorcycle', 'Bicycle']
    vehicles_bp = [x for x in vehicles_bp if (x.get_attribute('base_type') not in vehicles_to_remove)]

    # Get the map's spawn points
    spawn_points = world.get_map().get_spawn_points()
    # Keep only spawn points outside internal road
    spawn_points = list(filter(lambda trans: not is_within_area(trans.location, br, tl), spawn_points))

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    # Spawn num_vehicles vehicles randomly distributed throughout the map 
    # for each spawn point, we choose a random vehicle from the blueprint library
    batch = [SpawnActor(random.choice(vehicles_bp), random.choice(spawn_points)).then(SetAutopilot(FutureActor, True, tm.get_port())) for _ in range(num_vehicles)]
    

    results = client.apply_batch_sync(batch, True)

    if LOCAL:
        world.tick()
    else:
        # Wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.wait_for_tick()