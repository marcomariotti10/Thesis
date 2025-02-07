import os

def start_preprocessing(script_name):
    os.system(f'python {script_name} 4')

if __name__ == "__main__":
    scripts = [
        "save_positions.py",
        "conversion_3D_to_2,5D.py",
        "convertion_BB_into_2,5D.py",
        "eliminate_BB_without_points.py"
    ]

    for script in scripts:
        start_preprocessing(script)
