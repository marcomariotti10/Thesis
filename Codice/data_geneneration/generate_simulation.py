"""
CARLA Simulation Generator

This script creates and manages a CARLA simulation environment with vehicles and pedestrians.
It provides an interactive interface for controlling the simulation, spawning actors, and
managing simulation parameters. The script is designed for data generation in autonomous
driving research and thesis work.

Author: Marco Mariotti
Purpose: Generate simulation data for machine learning models in autonomous driving
"""

# Standard library imports
import glob
import os
import sys
import time
import logging
import argparse
import random
import json
import threading

# CARLA API setup - Add CARLA Python API to path
try:
    sys.path.append(glob.glob('C:/Users/marco/Desktop/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# CARLA imports
import carla
from carla import VehicleLightState as vls

# Local utility imports
from utility import *
from local_sim_utility import *
from spawn_objects import *

# =============================================================================
# SIMULATION CONFIGURATION PARAMETERS
# =============================================================================

LOCAL = True          # Whether to run in local synchronous mode
SEED = 42            # Random seed for reproducible simulations
PORT = 2000          # CARLA server port
PORTTM = 8000        # Traffic Manager port
IP = "127.0.0.1"     # Server IP address
N_WEHICLES = 40      # Number of vehicles to spawn
N_PED = 40           # Number of pedestrians to spawn

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_actor_blueprints(world, filter, generation):
    """
    Retrieve actor blueprints from CARLA world based on filter and generation.
    
    This function filters the available actor blueprints (vehicles, pedestrians, etc.)
    from the CARLA blueprint library based on the specified filter pattern and
    generation constraints.
    
    Args:
        world (carla.World): CARLA world instance
        filter (str): Filter pattern for blueprint selection (e.g., 'vehicle.*')
        generation (str): Generation constraint ('1', '2', or 'all')
        
    Returns:
        list: List of filtered blueprints matching the criteria
        
    Example:
        # Get all vehicle blueprints from generation 2
        vehicle_bps = get_actor_blueprints(world, 'vehicle.*', '2')
    """
    # Get blueprints from world library using the filter
    bps = world.get_blueprint_library().filter(filter)

    # If 'all' generations requested, return all filtered blueprints
    if generation.lower() == "all":
        return bps

    # If the filter returns only one blueprint, ignore generation filtering
    # and return that single blueprint
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is valid (1 or 2) and filter accordingly
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# =============================================================================
# MAIN SIMULATION EXECUTION
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # COMMAND LINE ARGUMENT PARSING
    # =========================================================================
    argparser = argparse.ArgumentParser(
        description=__doc__)
    
    # Vehicle filtering arguments
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
    
    # Pedestrian filtering arguments
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

    # =========================================================================
    # CARLA CLIENT SETUP AND CONNECTION
    # =========================================================================
    print("Connecting to server...")

    # Establish connection to CARLA server
    client = carla.Client(IP, PORT)
    client.set_timeout(15.0)
    world = client.get_world()

    try:

        # =====================================================================
        # TRAFFIC MANAGER AND WORLD SETTINGS CONFIGURATION
        # =====================================================================
        
        # Initialize Traffic Manager with specified port and seed
        tm = client.get_trafficmanager(PORTTM)
        tm.set_random_device_seed(SEED)  # Ensure reproducible behavior
        tm.set_global_distance_to_leading_vehicle(2.5)  # Set following distance
        
        # Configure world settings for local synchronous simulation
        if LOCAL:
            # Set fixed fps to 20 (0.05 seconds per tick)
            settings = world.get_settings()
            settings.synchronous_mode = True      # Enable synchronous mode
            tm.set_synchronous_mode(True)         # Sync traffic manager
            synchronous_master = True
            settings.fixed_delta_seconds = 0.05   # 20 FPS
            settings.no_rendering_mode = False    # Keep rendering enabled
            world.apply_settings(settings)

            # Reload world to apply settings without losing configuration
            client.reload_world(False) 
        
        # =====================================================================
        # ACTOR LISTS AND BLUEPRINT INITIALIZATION
        # =====================================================================
        
        # Initialize lists to keep track of spawned actors
        vehicles_list = []
        walkers_list = []
        all_id = []

        # Get vehicle and pedestrian blueprints based on command line arguments
        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        #blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 2]  # Uncomment for bikes only
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)
        blueprints = sorted(blueprints, key=lambda bp: bp.id)  # Sort for consistency

        # =====================================================================
        # SPAWN POINT VALIDATION AND PREPARATION
        # =====================================================================
        
        # Get available spawn points from the map
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        # Validate that we have enough spawn points for requested vehicles
        if N_WEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)  # Randomize spawn locations
        elif N_WEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, N_WEHICLES, number_of_spawn_points)
            N_WEHICLES = number_of_spawn_points  # Limit to available points

        # Define CARLA command types for batch operations
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # =====================================================================
        # PEDESTRIAN SPAWNING
        # =====================================================================
        print("Pedestrian creation...")
        # Spawn pedestrians using utility function and store their controllers
        controllers = spawn_pedestrians(client, N_PED, LOCAL)      

        # =====================================================================
        # VEHICLE SPAWNING
        # =====================================================================
        print("Vehicles creation...")
        
        # Prepare batch commands for efficient vehicle spawning
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= N_WEHICLES:
                break
                
            # Select random blueprint and configure attributes
            blueprint = random.choice(blueprints)
            
            # Set random color if available
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
                
            # Set random driver if available
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            
            # Set role name for identification
            blueprint.set_attribute('role_name', 'autopilot')

            # Create batch command: spawn vehicle and enable autopilot
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, tm.get_port())))

        # Execute batch spawning synchronously and handle responses
        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)
        # =====================================================================
        # SIMULATION SETUP AND USER INTERFACE
        # =====================================================================

        print("Setting up the simulation...")

        # Create informational image showing available commands
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

        # Initialize simulation state variables
        debug_draw_bb = False  # Debug bounding box visualization flag
        tour = False          # Tour mode flag
        
        # Display command window
        cv2.imshow("Command Window", image)
        print("Start the simulation!")
        
        # =====================================================================
        # MAIN SIMULATION LOOP
        # =====================================================================
        while True:

            # Advance simulation by one tick
            if LOCAL:
                world.tick()  # Synchronous mode: manually advance
            else:
                world.wait_for_tick()  # Asynchronous mode: wait for server

            # Handle debug visualization
            if debug_draw_bb:
                draw_debug(world, static_car_bboxes, static_bike_bboxes)
                
            # Handle tour mode
            if tour:
                tour, tour_t, current_position = perform_tour(world, tour_positions, tour_t, current_position)
                
            # ================================================================= 
            # KEYBOARD INPUT HANDLING
            # =================================================================
            key = cv2.waitKey(1)
            
            # Quit simulation
            if key == ord('q'):
                break
                
            # Move all pedestrians to station location
            if key == ord('s'):
                for ctr in controllers:
                    ctr.go_to_location(carla.Location(48.0,3.0,0.0))
                    
            # Move all pedestrians to university location
            if key == ord('u'):
                for ctr in controllers:
                    ctr.go_to_location(carla.Location(-16.45,-129.2,0.0))
                    
            # Move all pedestrians to central street location
            if key == ord('c'):
                for ctr in controllers:
                    ctr.go_to_location(carla.Location(-59.10,11.10,0.0))
                    
            # Toggle debug bounding box visualization
            if key == ord('d'):
                debug_draw_bb = not debug_draw_bb
                
            # Spawn a new vehicle at the station
            if key == ord('v'):
                spawn_vehicle_station(world, tm)
                
            # Spawn a new pedestrian at the station
            if key == ord('p'):
                ctr = spawn_pedestrian_station(client, LOCAL)
                if ctr is not None:
                    controllers.append(ctr)
                    
            # Toggle tour mode on/off
            if key == ord('t'):
                if tour == False:
                    tour_t, current_position = start_tour(world, tour_positions)
                tour = not tour

            # Small delay to prevent excessive CPU usage
            time.sleep(0.05)
    # =========================================================================
    # EXCEPTION HANDLING AND CLEANUP
    # =========================================================================
    except Exception as e: 
        print(e)
    finally:
        # Close OpenCV windows
        cv2.destroyAllWindows()

        # =====================================================================
        # WORLD SETTINGS RESTORATION
        # =====================================================================
        if LOCAL:
            # Restore original world settings
            settings=world.get_settings()
            settings.synchronous_mode = False     # Disable synchronous mode
            settings.no_rendering_mode = False    # Ensure rendering is enabled
            settings.fixed_delta_seconds = None   # Return to variable time step
            world.apply_settings(settings)
            
            # Restore traffic manager to asynchronous mode
            tm = client.get_trafficmanager(PORTTM)
            tm.set_synchronous_mode(False)
        
        # =====================================================================
        # ACTOR CLEANUP AND DESTRUCTION
        # =====================================================================
        
        # Destroy all sensors in the world
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*sensor*')])
        
        # Destroy all vehicles
        vehicles_list = world.get_actors().filter('*vehicle*')
        # First disable autopilot for all vehicles
        client.apply_batch([carla.command.SetAutopilot(x, False, PORT) for x in vehicles_list])
        # Then destroy all vehicles
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        
        # Destroy all pedestrians and their controllers
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*controller*')])
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*walker*')])
        
        # Destroy any static actors (if any were created)
        client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors().filter('*static*')])




