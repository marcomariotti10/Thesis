import os
import subprocess

def start_preproces(script_name, user_input):
    process = subprocess.Popen(['python', script_name], stdin=subprocess.PIPE)
    process.communicate(input=user_input.encode())  # Provide the required input

if __name__ == "__main__":
    scripts = [
        #"fit_entire_dataset.py",
        #"separation_train_val_test.py",
        "generate_chuncks.py",
        #"generate_augmentation.py",
        #"combine_augmentation.py",
        "generate_ffcv.py"
    ]

    current_dir = os.path.dirname(__file__)

    for script in scripts:
        new_dir = os.path.join(current_dir, script)
        print(f"Starting process: {script}")
        start_preproces(new_dir, '4\n') 
        print("\n")