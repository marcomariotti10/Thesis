import os
import subprocess

def start_preproces(script_name, user_input):
    process = subprocess.Popen(['python', script_name], stdin=subprocess.PIPE)
    process.communicate(input=user_input.encode())  # Provide the required input

if __name__ == "__main__":
    scripts = [
        "generate_chuncks.py",
        "augmentation.py",
        "generate_ffcv.py",
        "NN_for_server_ffcv.py",
    ]

    current_dir = os.path.dirname(__file__)

    for script in scripts:
        new_dir = os.path.join(current_dir, script)
        print(f"Starting process: {script}")
        start_preproces(new_dir, '4\n') 
        print("\n")