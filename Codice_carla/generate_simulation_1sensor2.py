import carla
import random
import json
import threading
import subprocess

from utility import *
from local_sim_utility import *
from spawn_objects import *

LOCAL = True
SEED = 42
PORT = 2000
IP = "127.0.0.1"

if __name__ == "__main__":

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

    try:

        tm = client.get_trafficmanager(PORT)

        # ---------------------
        # STATIC SENSORS
        # ---------------------
        print("Static sensors creation...")
        threads = []
        sensors = []
        static = []
        spawn_static_sensors(sensors_config2, sensors, static, world, LOCAL)
 
        
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
                process.terminate()
                process.wait()  # Wait for save_BB.py to exit cleanly
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