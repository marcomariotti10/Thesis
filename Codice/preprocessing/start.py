import os
import subprocess

def start_preprocessing(script_name, user_input):
    process = subprocess.Popen(['python', script_name], stdin=subprocess.PIPE)
    process.communicate(input=user_input.encode())  # Provide the required input

if __name__ == "__main__":
    scripts = [
        "syncronization.py",
        "LiDAR_grid.py",
        "occupancy_grid.py",
        "filter_grid.py"
    ]

    current_dir = os.path.dirname(__file__)

    for script in scripts:
        new_dir = os.path.join(current_dir, script)
        print(f"Starting preprocessing for {script}")
        start_preprocessing(new_dir, 'all\n') 
        print("\n")