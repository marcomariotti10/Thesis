import carla
import random
import json
import threading
import glob
import os
import sys
import time
import logging
import argparse

from utility import *
from local_sim_utility import *
from spawn_objects import *

LOCAL = True
SEED = 42
PORT = 2000
IP = "127.0.0.1"
N_VEHICLES = 30
N_PED = 20

def get_spawn_points(world, delimited_region):
    spawn_points = world.get_map().get_spawn_points()
    filtered_spawn_points = [point for point in spawn_points if not is_within_region(point, delimited_region)]
    return filtered_spawn_points

def is_within_region(point, region):
    # Implement your logic to check if the point is within the delimited region
    # For example, region could be a bounding box with min and max coordinates
    min_x, min_y, max_x, max_y = region
    return min_x <= point.location.x <= max_x and min_y <= point.location.y <= max_y

def respawn_vehicles(world, client, tm, vehicles_list, N_VEHICLES, delimited_region):
    current_vehicles = world.get_actors().filter('vehicle.*')
    if len(current_vehicles) < N_VEHICLES:
        spawn_points = get_spawn_points(world, delimited_region)
        for _ in range(N_VEHICLES - len(current_vehicles)):
            spawn_point = random.choice(spawn_points)
            blueprint = random.choice(world.get_blueprint_library().filter('vehicle.*'))
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                vehicles_list.append(vehicle)

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')

    args = argparser.parse_args()

    f = open(CONFIG_DIR + '/map_03_final/lidar1_map03.json')
    sensors_config1 = json.load(f)
    random.seed(SEED)
    
    f = open(CONFIG_DIR + '/map_03_final/lidar5_map03.json')
    sensors_config2 = json.load(f)
    random.seed(SEED)
    
    f = open(CONFIG_DIR + '/map_03_final/lidar6_map03.json')
    sensors_config3 = json.load(f)
    random.seed(SEED)

    print("Connecting to server...")

    client = carla.Client(IP, PORT)
    client.set_timeout(150.0)
    world = client.get_world()
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        
        world = client.get_world()

        tm = client.get_trafficmanager(PORT)
        tm.set_global_distance_to_leading_vehicle(2.5)
        if LOCAL:
            # Set fixed fps to 20
            settings = world.get_settings()
            
            tm.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False

            world.apply_settings(settings)

            client.reload_world(False) # reload map keeping the world settings
        
        vehicles_list = []
        walkers_list = []
        all_id = []

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if N_VEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif N_VEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, N_VEHICLES, number_of_spawn_points)
            N_WEHICLES = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor  

        # ---------------------
        # VEHICLES
        # ---------------------
        print("Vehicles creation...")
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= N_WEHICLES:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, tm.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
        
        # ---------------------
        # PEDESTRIANS
        # ---------------------
        print("Pedestrian creation...")
        controllers = spawn_pedestrians(client, 20, LOCAL)    
        
        # -------------------
        # SIMULATION
        # -------------------

        print('spawned %d vehicles and pedestrians, press Ctrl+C to exit.' % (len(vehicles_list)))

        # Example of how to use Traffic Manager parameters
        tm.global_percentage_speed_difference(30.0)

        delimited_region = (-130, -30, -40, 60)  # Define your delimited region coordinates

        while True:
            if synchronous_master:
                respawn_vehicles(world, client, tm, vehicles_list, N_VEHICLES, delimited_region)
                world.tick()
            else:
                world.wait_for_tick()

    finally:

        if synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        all_actors = world.get_actors(all_id)
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)



