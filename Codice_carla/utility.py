import carla
import cv2
import numpy as np

import math
from transforms3d.euler import euler2quat

HOME = r'C:/Users/Airlab/Desktop/Tesi/Codice_carla'   # <---- CHANGE THIS
SIMULATION_DIR = HOME + r'/simulation_data'
CONFIG_DIR = HOME + r'/config'

def convert_to_usable_resolution(camera_resolution):
    camera_resolution['image_size_x'] = str(round(camera_resolution['image_size_x']/256)*256)
    camera_resolution['image_size_y'] = str(round(camera_resolution['image_size_y']/256)*256)
    return camera_resolution

def is_within_area(location, area_br, area_tl):
    """
    Define the coordinates of the area in this way:
    area_br = {
        'x': -16.700000,
        'y': -15.500000
    }
    area_tl = {
        'x': -221.100000,
        'y': 145.250000
    }
    :param area_bl: bottom left point
    :param area_tr: top right point
    """
    return location.x < area_br['x'] and location.y > area_br['y'] and location.x > area_tl['x'] and location.y < area_tl['y']


def create_text_image(message):
    """
    Generate a white image with a message in the middle to be used with cv2
    :param message: text message to display (can contain multiple lines delimited by '\n')
    :return: the generated image
    """
    padding = 50
    interline = 10
    lines = message.split('\n')  # Split the message into lines
    line_height = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][1] + interline # Get the height of a line
    image_height = line_height * len(lines) + padding  # Add extra padding at the top and bottom of the text

    max_line_width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0] for line in lines)
    image_width = max(400, max_line_width + padding)  # Set a minimum width of 400 and add extra padding to the sides

    image = 255 * np.ones((image_height, image_width, 3), dtype=np.uint8)
    text_color = (0, 0, 0)  # Black color
    text_thickness = 1
    font_scale = 0.7

    # Calculate the starting position for the first line
    y = (image.shape[0] - (line_height * len(lines)))
    x = padding//2

    # Draw each line of text
    for line in lines:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        #x = (image.shape[1] - text_size[0]) // 2  # Calculate the x position for center alignment
        text_position = (x, y)
        cv2.putText(image, line, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)
        y += line_height  # Move to the next line

    return image

def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple

    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).

    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)

    return (roll, pitch, yaw)

def carla_rotation_to_quat(carla_rotation):
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    quat = euler2quat(roll, pitch, yaw)
    return quat


def draw_debug(world, static_car_bboxes, static_bike_bboxes):
    car_color = carla.Color(0,0,255)
    ped_color = carla.Color(0,255,0)
    bike_color = carla.Color(255,0,0)
    
    settings = world.get_settings()
    bbox_life_time = settings.fixed_delta_seconds
    if bbox_life_time == None:
        bbox_life_time = 0.06
    else:
        bbox_life_time += 0.01 # Make the life time slightly longer than the fixed delta seconds to make it visible

    for vehicle in world.get_actors().filter('vehicle.*'):
        # Draw bounding box
        transform = vehicle.get_transform()
        bounding_box = vehicle.bounding_box
        if int(vehicle.attributes['number_of_wheels']) == 2:
            # Bikes actually don't have a bounding box and every class place the box in a different way, to solve this the bbox are drawn manually with heuristic values
            bounding_box.location = transform.location
            bounding_box.location += carla.Vector3D(0.3,0.2,0.7)
            bounding_box.extent = carla.Vector3D(0.6,0.4,0.7)
            world.debug.draw_box(bounding_box, transform.rotation,  thickness=0.05, color=bike_color, life_time=bbox_life_time)
        else:
            bounding_box.location += transform.location
            world.debug.draw_box(bounding_box, transform.rotation, thickness=0.05, color=car_color, life_time=bbox_life_time)

    for bbox in static_car_bboxes:
        world.debug.draw_box(carla.BoundingBox(bbox.location, bbox.extent), bbox.rotation, thickness=0.05, color=car_color, life_time=bbox_life_time)
    for bbox in static_bike_bboxes:
        world.debug.draw_box(carla.BoundingBox(bbox.location, bbox.extent), bbox.rotation, thickness=0.05, color=bike_color, life_time=bbox_life_time)

    for pedestrian in world.get_actors().filter('walker.*'):
        # Draw bounding box
        transform = pedestrian.get_transform()
        bounding_box = pedestrian.bounding_box
        bounding_box.location += transform.location
        world.debug.draw_box(bounding_box, transform.rotation, thickness=0.05, color=ped_color, life_time=bbox_life_time)


def u_to_c_loc(X,Y,Z):
    '''Utility function to easily copy coordinates from unreal engine to carla'''
    return (carla.Location(X/100,Y/100,Z/100))

def u_to_c_rot(Pitch, Yaw, Roll):
    '''Utility function to easily copy rotations from unreal engine to carla'''
    return carla.Rotation(Pitch,Yaw,Roll)

tour_positions = [
    carla.Transform(u_to_c_loc(X=-13675.000000,Y=-4790.000000,Z=714.000000), u_to_c_rot(Pitch=0.000000,Yaw=-66.130859,Roll=0.000000)),
    carla.Transform(carla.Location(-108.96302734,-47.69933105,6.74400269), carla.Rotation()),
    carla.Transform(carla.Location(-62.16302734,-30.09933105,1.94400269), carla.Rotation()),
    carla.Transform(carla.Location(-15.91302734,-20.14933594,1.34400269), carla.Rotation()),
    carla.Transform(u_to_c_loc(X=1750.000000,Y=-1690.000000,Z=404.000000), u_to_c_rot(Pitch=-5.864746,Yaw=31.253325,Roll=-2.580505)), #piu alto
    carla.Transform(u_to_c_loc(X=2330.000000,Y=1265.000000,Z=404.000000), u_to_c_rot(Pitch=-5.864746,Yaw=46.135742,Roll=-2.580505)), # più staccato
    carla.Transform(u_to_c_loc(X=3260.000000,Y=6045.000000,Z=404.000000), u_to_c_rot(Pitch=-5.864746,Yaw=46.135742,Roll=-2.580505)), # più avanti
    carla.Transform(u_to_c_loc(X=4125.000000,Y=10180.000000,Z=404.000000), u_to_c_rot(Pitch=-5.864746,Yaw=31.253325,Roll=-2.580505)),
]
acutal_position = 0
tour_t = 0

def carla_loc_lerp(pos1, pos2, t):
    return carla.Location((1 - t) * pos1.x + t * pos2.x,
                          (1 - t) * pos1.y + t * pos2.y,
                          (1 - t) * pos1.z + t * pos2.z)

def carla_rot_lerp(pos1, pos2, t):
    return carla.Rotation((1 - t) * pos1.pitch + t * pos2.pitch,
                          (1 - t) * pos1.yaw + t * pos2.yaw,
                          (1 - t) * pos1.roll + t * pos2.roll)

def carla_trans_lerp(pos1, pos2, t):
    return carla.Transform(carla_loc_lerp(pos1.location, pos2.location, t), 
                           carla_rot_lerp(pos1.rotation, pos2.rotation,t))

def perform_tour(world, tour_positions, tour_t, current_position):
    spectator = world.get_spectator()

    settings = world.get_settings()
    dt = settings.fixed_delta_seconds
    if dt == None:
        dt = 0.06
    
    tour_t += dt*0.3
    if tour_t > 1:
        current_position+=1
        print("Checkpoint number ", current_position)
        if current_position >= len(tour_positions)-1:
            return False, 0, 0
        tour_t=0
        spectator.set_transform(tour_positions[current_position])
        return True, tour_t, current_position
        
    spectator.set_transform(carla_trans_lerp(tour_positions[current_position], tour_positions[(current_position+1)], tour_t))

    return True, tour_t, current_position

    
def start_tour(world, tour_positions):
    spectator = world.get_spectator()
    spectator.set_transform(tour_positions[0])
    tour_t = 0
    current_position = 0
    print("Tour started!")

    return tour_t, current_position