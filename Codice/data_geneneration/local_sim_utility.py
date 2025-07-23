import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image
import os
import open3d as o3d
from matplotlib import cm
import time
import cv2
import carla

from utility import create_text_image
from config import *

def save_image(image, directory):
    """
    This function, given as input a carla.Image object, it saves the image in jpeg format to the
    specified directory. If the directory does not exists, this function will create it.
    This function assumes that the image is BGRA: it will drop the alpha channel and revert it to 
    RGB format.
    :param image: carla.Image object to save to disk.
    :param directory: relative path (with respect to the SIMULATION_DIR directory) where to store the image.
    :return None.
    """
    raw_data = image.raw_data
    raw_data = np.reshape(raw_data, (image.height, image.width, 4))
    raw_data = raw_data[:, :, :3] # drop alpha channel
    raw_data = raw_data[:,:,::-1] # convert BRG to RGB
    raw_data = Image.fromarray(raw_data)

    # save the Image
    if not os.path.exists(f'{SIMULATION_DIR}/{directory}/'):
        os.makedirs(f'{SIMULATION_DIR}/{directory}/')
    raw_data.save(f'{SIMULATION_DIR}/{directory}/{image.frame}.jpeg')


def save_semseg(image, directory):
    image.save_to_disk(f'{SIMULATION_DIR}/{directory}/{image.frame}.jpg', carla.ColorConverter.CityScapesPalette)


def save_depth(image, directory, depth_attributes):
    """
    This function, given as input a carla.Image object, it saves the image in jpeg format to the
    specified directory. If the directory does not exists, this function will create it.
    This function assumes that the image is BGRA: it will drop the alpha channel and revert it to 
    RGB format. Then apply the conversion to gray scale distance.
    :param image: carla.Image object to save to disk in depth form.
    :param directory: relative path (with respect to the SIMULATION_DIR directory) where to store the image.
    :return None.
    """
    raw_data = image.raw_data
    raw_data = np.reshape(raw_data, (image.height, image.width, 4))
    raw_data = raw_data[:, :, :3] # drop alpha channel
    raw_data = raw_data[:,:,::-1] # convert BRG to RGB
    normalized = (raw_data[:,:,0] + raw_data[:,:,1] * 256.0 + raw_data[:,:,2] * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
    in_meters:np.ndarray = 1000 * normalized
    in_meters = np.clip(in_meters, depth_attributes['min_depth'], depth_attributes['max_depth'])
    val = np.log1p(in_meters) / np.log1p(depth_attributes['max_depth']) * 255
    raw_data = np.repeat(val[:, :, np.newaxis], 3, axis=2)
    raw_data = Image.fromarray(raw_data.astype(np.uint8))


    # save the Image
    if not os.path.exists(f'{SIMULATION_DIR}/{directory}/'):
        os.makedirs(f'{SIMULATION_DIR}/{directory}/')
    raw_data.save(f'{SIMULATION_DIR}/{directory}/{image.frame}.jpeg')


    # Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

def lidar_callback(point_cloud, point_list):
    """
    Prepares a point cloud with intensity
    colors ready to be consumed by Open3D
    """
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    points = data[:, :-1]

    points[:, :1] = -points[:, :1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)

def false_lidar_cb(point_cloud):
    pass


def show_o3d_lidar_visualizer(lidar, point_list):
    # Open3D visualiser for LIDAR
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Static Lidar',
        width=960,
        height=540,
        left=480,
        top=270)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    add_open3d_axis(vis)

    image = create_text_image("Press q to close")

    try: 
        # Show the image in a window
        cv2.imshow("Command Window", image)
        # Update geometry in game loop
        frame = 0
        while True:
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)
            
            vis.poll_events()
            vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            frame += 1

            # Break if user presses 'q'
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            # add other commands here

    finally:
        cv2.destroyAllWindows()
        lidar.stop()
        lidar.destroy()
        vis.destroy_window()