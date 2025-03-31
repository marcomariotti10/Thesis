import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import datetime
import sys
import os
import shutil
import csv
import numpy as np
import pandas as pd
import threading
from multiprocessing import Pool

def process_file(file_path):
    """Function to process a single file"""
    print(f"Processing: {file_path}")

def process_sensor(sensor_id):
    """Creates threads for processing files in a sensor"""
    print(f"Processing Sensor {sensor_id}...")

    sensor_directory = LIDAR_DIRECTORY.replace("X", str(sensor_id))
    files = [os.path.join(sensor_directory, f) for f in os.listdir(sensor_directory)]
    
    threads = []
    for file in files:
        thread = threading.Thread(target=process_file, args=(file,))
        threads.append(thread)
        thread.start()
        
        # Limit active threads to avoid system overload
        if len(threads) >= 1000:  # Adjust this limit based on your system
            for t in threads:
                t.join()
            threads = []  # Reset thread list

    # Ensure remaining threads finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    sensor_threads = []
    for sensor_id in range(1, NUMBER_OF_SENSORS + 1):
        thread = threading.Thread(target=process_sensor, args=(sensor_id,))
        sensor_threads.append(thread)
        thread.start()
    
    for thread in sensor_threads:
        thread.join()

    print("Processing completed for all sensors.")