import os
import subprocess

def start_preprocessing(script_name, user_input):
    process = subprocess.Popen(['python', script_name], stdin=subprocess.PIPE)
    process.communicate(input=user_input.encode())  # Provide the required input

if __name__ == "__main__":
    scripts = [
        "save_positions.py",
        "conversion_3D_to_2,5D.py",
        "convertion_BB_into_2,5D.py",
        "eliminate_BB_without_points.py"
    ]

    current_dir = os.path.dirname(__file__)

    for script in scripts:
        new_dir = os.path.join(current_dir, script)
        print(f"Starting preprocessing for {script}")
        start_preprocessing(new_dir, '4\n')  # Provide '1' as input for all scripts
        print("\n")