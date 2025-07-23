import os
import subprocess

def start_preproces(script_path, args):
    # Start the process and wait for it to finish
    process = subprocess.Popen(['python', script_path] + args)
    process.wait()  # Wait for the current process to finish

if __name__ == "__main__":
    model_types = [
        "MultiHeadUNetAutoencoder",
        "MultiHeadCBAMAutoencoder",
        "MultiHeadAutoencoder"
    ]

    script_name = "train_encoder.py"
    current_dir = os.path.dirname(__file__)
    script_path = os.path.join(current_dir, script_name)

    for model_type in model_types:
        args = [f"--model_type={model_type}"]
        print(f"Starting process: {script_name} {args}")
        start_preproces(script_path, args)
        print(f"Finished training with model_type={model_type}\n")
