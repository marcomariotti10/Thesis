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
import carla
import random
import time
import logging
import argparse

LOCAL = True

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

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

    print("Connecting to server...")

    client = carla.Client(IP, SERVER_PORT)
    client.set_timeout(15.0)
    world = client.get_world()

    try:

        tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
        tm.set_random_device_seed(SEED)
        tm.set_global_distance_to_leading_vehicle(2.5)
        if LOCAL:
            # Set fixed fps to 20
            settings = world.get_settings()
            settings.synchronous_mode = True
            tm.set_synchronous_mode(True)
            synchronous_master = True
            settings.fixed_delta_seconds = 0.05
            settings.no_rendering_mode = False
            world.apply_settings(settings)

            client.reload_world(False) # reload map keeping the world settings
        
        vehicles_list = []
        walkers_list = []
        all_id = []

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if NUMBER_VEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif NUMBER_VEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, NUMBER_VEHICLES, number_of_spawn_points)
            NUMBER_VEHICLES = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor


        # ---------------------
        # PEDESTRIANS
        # ---------------------
        print("Pedestrian creation...")
        controllers = spawn_pedestrians(client, NUMBER_PEDESTRIANS, LOCAL)      

        # ---------------------
        # VEHICLES
        # ---------------------
        print("Vehicles creation...")
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= NUMBER_VEHICLES:
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
        

        # -------------------
        # SIMULATION
        # -------------------

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
    finally:
        if LOCAL:
            settings=world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)
            tm.set_synchronous_mode(False)
        
        # Destroy sensors
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*sensor*')])
        # Destroy vehicles
        vehicles_list = world.get_actors().filter('*vehicle*')
        client.apply_batch([carla.command.SetAutopilot(x, False, SERVER_PORT) for x in vehicles_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        # Destroy pedestrians
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*controller*')])
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*walker*')])
         # Destroy sensors home actors
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*static*')])




