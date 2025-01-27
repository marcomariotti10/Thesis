import sys
import os

def link_constants():
    # Dynamically construct the path to the data_gen_and_processing folder
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    data_gen_and_processing_dir = os.path.join(parent_dir, 'data_gen_and_processing')

    # Add the path to the constants module
    sys.path.insert(0, data_gen_and_processing_dir)

    # Print paths for debugging
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Data gen and processing directory: {data_gen_and_processing_dir}")
    
    # Add the path to the constants module
    sys.path.append(data_gen_and_processing_dir)
