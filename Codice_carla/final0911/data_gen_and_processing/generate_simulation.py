import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('C:/Users/marco/Desktop/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import VehicleLightState as vls
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
N_WEHICLES = 30
N_PED = 20

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
    client.set_timeout(15.0)
    world = client.get_world()

    try:

        tm = client.get_trafficmanager(PORT)
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

        if N_WEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif N_WEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, N_WEHICLES, number_of_spawn_points)
            N_WEHICLES = number_of_spawn_points

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor


        # ---------------------
        # PEDESTRIANS
        # ---------------------
        print("Pedestrian creation...")
        controllers = spawn_pedestrians(client, 20, LOCAL)      

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
        

        # -------------------
        # SIMULATION
        # -------------------

        print("Setting up the simulation...")

        image = create_text_image("Commands:\n" + \
                                " Press Q to close and destroy all the actors\n" + \
                                " Press S to move pedestrians to station\n" + \
                                " Press U to move pedestrians to university\n" + \
                                " Press C to move pedestrians to central street\n" + \
                                " Press V to spawn a vehicle in front of the station\n" + \
                                " Press P to spawn a pedestrian in front of the station\n" + \
                                " Press D to show/hide DebugBBox\n" + \
                                " Press T to start/stop a Tour of the map"
                                    )

        debug_draw_bb = False
        tour = False
        # Show the image in a window
        cv2.imshow("Command Window", image)
        print("Start the simulation!")
        while True:

            if LOCAL:
                world.tick()
            else:
                world.wait_for_tick()

            if debug_draw_bb:
                draw_debug(world, static_car_bboxes, static_bike_bboxes)
            if tour:
                tour, tour_t, current_position = perform_tour(world, tour_positions, tour_t, current_position)
            # Break if user presses 'q'
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            # add other commands here
            if key == ord('s'):
                for ctr in controllers:
                    ctr.go_to_location(carla.Location(48.0,3.0,0.0))
            if key == ord('u'):
                for ctr in controllers:
                    ctr.go_to_location(carla.Location(-16.45,-129.2,0.0))
            if key == ord('c'):
                for ctr in controllers:
                    ctr.go_to_location(carla.Location(-59.10,11.10,0.0))
            if key == ord('d'):
                debug_draw_bb = not debug_draw_bb
            if key == ord('v'):
                spawn_vehicle_station(world, tm)
            if key == ord('p'):
                ctr = spawn_pedestrian_station(client, LOCAL)
                if ctr is not None:
                    controllers.append(ctr)
            if key == ord('t'):
                if tour == False:
                    tour_t, current_position = start_tour(world, tour_positions)
                tour = not tour

            time.sleep(0.05)
    except Exception as e: 
        print(e)
    finally:
        cv2.destroyAllWindows()

        if LOCAL:
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)
        
        # Destroy sensors
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*sensor*')])
        # Destroy vehicles
        vehicles_list = world.get_actors().filter('*vehicle*')
        client.apply_batch([carla.command.SetAutopilot(x, False, PORT) for x in vehicles_list])
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        # Destroy pedestrians
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*controller*')])
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*walker*')])
         # Destroy sensors home actors
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*static*')])




