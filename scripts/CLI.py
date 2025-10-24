#!/usr/bin/env python3
"""
Comprehensive CLI Framework for Thesis Project
This script provides a command-line interface to control the entire framework.
"""

import os
import sys
import subprocess
from pathlib import Path
from settings import *

# Add the src directory to Python path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Import configuration constants
from config import *


class FrameworkCLI:
    def __init__(self):
        self.src_dir = src_dir
        self.script_dir = script_dir
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the application header"""
        print("=" * 60)
        print("                 FRAMEWORK CONTROL CENTER")
        print("=" * 60)
        print()
    
    def print_main_menu(self):
        """Print the main menu options"""
        print("Please select an option:")
        print("1. Generate Data")
        print("2. Preprocess Data")
        print("3. Train Neural Network Model")
        print("4. Visualization & Results")
        print("9. Exit")
        print()
    
    def print_data_generation_menu(self):
        """Print data generation submenu"""
        print("\nData Generation Options:")
        print("1. Start Simulation")
        print("2. Start Saving Simulation Snapshots")
        print("3. Start Sensor")
        print("9. Back to Main Menu")
        print()
    
    def print_preprocessing_menu(self):
        """Print preprocessing submenu"""
        print("\nPreprocessing Options:")
        print("1. Run All Preprocessing Steps")
        print("2. Run Only Sample Generation Steps")
        print("9. Back to Main Menu")
        print()
    
    def print_training_menu(self):
        """Print training submenu"""
        print("\nTraining Options:")
        print("1. Train Autoencoder")
        print("2. Train Diffusion Model")
        print("9. Back to Main Menu")
        print()
    
    def print_visualization_menu(self):
        """Print visualization submenu"""
        print("\nVisualization & Results Options:")
        print("1. Plot Autoencoder Results")
        print("2. Plot Diffusion Results")
        print("3. Plot Grid Map")
        print("4. Plot Point Cloud")
        print("9. Back to Main Menu")
        print()
    
    def get_user_input(self, prompt="Enter your choice: "):
        """Get user input with error handling"""
        try:
            return input(prompt).strip()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)
    
    def run_script(self, script_path, args=None, cwd=None):
        """Run a Python script with optional arguments"""
        try:
            cmd = [sys.executable, str(script_path)]
            if args:
                cmd.extend(args)
            
            print("-" * 40)
            
            if cwd:
                result = subprocess.run(cmd, cwd=str(cwd), check=True)
            else:
                result = subprocess.run(cmd, check=True)
                
            print("-" * 40)
            print("Process completed successfully.")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError running script: {e}")
        except FileNotFoundError:
            print(f"\nError: Script not found at {script_path}")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
    
    def handle_data_generation(self):
        """Handle data generation options"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_data_generation_menu()
            
            choice = self.get_user_input()
            
            if choice == "1":
                # Start simulation
                script_path = self.src_dir / "data_generation" / "simulation.py"
                self.run_script(script_path)
                input("\nPress Enter to continue...")
                
            elif choice == "2":
                # Start snapshots
                script_path = self.src_dir / "data_generation" / "snapshots.py"
                self.run_script(script_path)
                input("\nPress Enter to continue...")
                
            elif choice == "3":
                # Start sensor
                sensor_num = self.get_sensor_number()
                if sensor_num:
                    print(f"\nStarting sensor {sensor_num}...")
                    script_path = self.src_dir / "data_generation" / "sensors.py"
                    try:
                        self.run_script(script_path, args=["--number", str(sensor_num)])
                    except Exception as e:
                        print(f"Error running sensor {sensor_num}: {e}")
                        print("Make sure CARLA is running and all dependencies are installed.")
                input("\nPress Enter to continue...")
                
            elif choice == "9":
                # Back to main menu
                break
                
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")
    
    def get_sensor_number(self):
        """Get sensor number from user with validation"""
        while True:
            try:
                print(f"\nAvailable sensors: 1 to {NUMBER_OF_SENSORS}")
                sensor_num = int(self.get_user_input("Enter sensor number: "))
                
                if 1 <= sensor_num <= NUMBER_OF_SENSORS:
                    # Verify the sensor file exists
                    sensor_file = self.script_dir / "sensors_specs" / f"lidar{sensor_num}.json"
                    if sensor_file.exists():
                        return sensor_num
                    else:
                        print(f"Error: Sensor specification file not found: {sensor_file}")
                        print("Please check that the sensor files are properly installed.")
                else:
                    print(f"Invalid sensor number. Please enter a number between 1 and {NUMBER_OF_SENSORS}.")
                    
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except KeyboardInterrupt:
                return None
    
    def get_starting_file_number(self, directory_path, file_extension='.ply'):
        """Get starting file number from user with validation"""
        try:
            # Count available files in the directory
            import os
            files = sorted([f for f in os.listdir(directory_path) if f.endswith(file_extension)])
            max_files = len(files)
            
            if max_files == 0:
                print(f"No {file_extension} files found in directory: {directory_path}")
                return None
                
            print(f"Available files: 1 to {max_files}")
            
            while True:
                try:
                    file_num = int(self.get_user_input("Enter starting file number: "))
                    file_num = file_num - 1  # Convert to zero-based index
                    
                    if 0 <= file_num < max_files:
                        return file_num
                    else:
                        print(f"Invalid file number. Please enter a number between 1 and {max_files}.")
                        
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
                except KeyboardInterrupt:
                    return None
                    
        except FileNotFoundError:
            print(f"Directory not found: {directory_path}")
            return None
        except Exception as e:
            print(f"Error accessing directory: {e}")
            return None
    
    def get_model_size(self):
        """Get model size from user for autoencoder training"""
        while True:
            try:
                print("\nSelect autoencoder model size:")
                print("1. Small")
                print("2. Medium") 
                print("3. Large")
                print()
                choice = self.get_user_input("Enter your choice (1-3): ")
                
                if choice == "1":
                    return "small"
                elif choice == "2":
                    return "medium"
                elif choice == "3":
                    return "large"
                else:
                    print("Invalid choice. Please select 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                return None
    
    def get_autoencoder_plot_type(self):
        """Get plot type for autoencoder visualization"""
        while True:
            try:
                print("\nSelect autoencoder plot type:")
                print("1. Plot Inference")
                print("2. Plot Metrics")
                print()
                choice = self.get_user_input("Enter your choice (1-2): ")
                
                if choice == "1":
                    return "inference"
                elif choice == "2":
                    return "metrics"
                else:
                    print("Invalid choice. Please select 1 or 2.")
                    
            except KeyboardInterrupt:
                return None
    
    def get_model_name(self):
        """Get model name from user"""
        while True:
            try:
                model_name = self.get_user_input("Enter the model name: ").strip()
                if model_name:
                    return model_name
                else:
                    print("Model name cannot be empty. Please try again.")
                    
            except KeyboardInterrupt:
                return None
    
    def extract_model_type_from_name(self, model_name):
        """Extract model type from model name"""
        # Common patterns in model names
        if "Small" in model_name or "small" in model_name:
            return "SmallModel"
        elif "Medium" in model_name or "medium" in model_name:
            return "MediumModel"
        elif "Large" in model_name or "large" in model_name or "Big" in model_name or "big" in model_name:
            return "LargeModel"
        else:
            # Default fallback - let user know we're using medium
            print("Could not detect model type from name. Using MediumModel as default.")
            return "MediumModel"

    def run_preprocessing_sequence(self, scripts):
        """Run a sequence of preprocessing scripts"""
        
        for i, (script_name, script_path) in enumerate(scripts, 1):
            print(f"\nStep {i}/{len(scripts)}: {script_name}")
            
            try:
                self.run_script(script_path)
            except Exception as e:
                print(f"✗ Failed: {script_name} - {e}")

    def handle_preprocessing(self):
        """Handle preprocessing options"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_preprocessing_menu()
            
            choice = self.get_user_input()
            
            if choice == "1":
                # Run all preprocessing steps in correct order
                scripts = [
                    ("Synchronize", self.src_dir / "preprocessing" / "sync_data.py"),
                    ("Convert Point Cloud", self.src_dir / "preprocessing" / "convert_pointcloud.py"),
                    ("Convert Snapshots", self.src_dir / "preprocessing" / "convert_snapshot.py"),
                    ("Filtering Snapshot", self.src_dir / "preprocessing" / "filtering_snapshot.py"),
                    ("Fit Scaler", self.src_dir / "preprocessing" / "fit_scaler.py"),
                    ("Shard Dataset", self.src_dir / "preprocessing" / "shard_dataset.py"),
                    ("Augment Combine", self.src_dir / "preprocessing" / "augment_combine.py"),
                    ("Build FFCV", self.src_dir / "preprocessing" / "build_ffcv.py")
                ]
                self.run_preprocessing_sequence(scripts)
                input("\nPress Enter to continue...")
                
            elif choice == "2":
                # Run only dataset division steps
                scripts = [
                    ("Shard Dataset", self.src_dir / "preprocessing" / "shard_dataset.py"),
                    ("Augment Combine", self.src_dir / "preprocessing" / "augment_combine.py"),
                    ("Build FFCV", self.src_dir / "preprocessing" / "build_ffcv.py")
                ]
                self.run_preprocessing_sequence(scripts)
                input("\nPress Enter to continue...")
                
            elif choice == "9":
                # Back to main menu
                break
                
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")
    
    def handle_training(self):
        """Handle neural network training options"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_training_menu()
            
            choice = self.get_user_input()
            
            if choice == "1":
                # Train autoencoder
                model_size = self.get_model_size()
                if model_size:
                    print(f"\nStarting autoencoder training with {model_size} model...")
                    script_path = self.src_dir / "models" / "autoencoder.py"
                    try:
                        self.run_script(script_path, args=["--model_size", model_size])
                    except Exception as e:
                        print(f"Error training autoencoder: {e}")
                        print("Make sure all dependencies are installed and data is prepared.")
                input("\nPress Enter to continue...")
                
            elif choice == "2":
                # Train diffusion model
                script_path = self.src_dir / "models" / "diffusion.py"
                self.run_script(script_path)
                input("\nPress Enter to continue...")
                
            elif choice == "9":
                # Back to main menu
                break
                
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")
    
    def handle_visualization(self):
        """Handle visualization and results options"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_visualization_menu()
            
            choice = self.get_user_input()
            
            if choice == "1":
                # Plot autoencoder results
                plot_type = self.get_autoencoder_plot_type()
                if plot_type:
                    model_name = self.get_model_name()
                    if model_name:
                        model_type = self.extract_model_type_from_name(model_name)
                        
                        script_path = self.src_dir / "viz" / "plot_autoencoder.py"
                        try:
                            args = ["--plot_type", plot_type, "--model_name", model_name, "--model_type", model_type]
                            self.run_script(script_path, args=args)
                        except Exception as e:
                            print(f"Error running autoencoder visualization: {e}")
                input("\nPress Enter to continue...")
                
            elif choice == "2":
                # Plot diffusion results
                model_name = self.get_model_name()
                if model_name:
                    script_path = self.src_dir / "viz" / "plot_diffusion.py"
                    try:
                        self.run_script(script_path, args=["--model_name", model_name])
                    except Exception as e:
                        print(f"Error running diffusion visualization: {e}")
                input("\nPress Enter to continue...")
                
            elif choice == "3":
                # Plot grid map
                print("\n" + "="*40)
                print("GRID MAP VISUALIZATION")
                print("="*40)
                
                # Get sensor number
                sensor_num = self.get_sensor_number()
                if sensor_num is None:
                    input("\nPress Enter to continue...")
                    continue
                
                # Get directory path
                directory_path = HEIGHTMAP_DIRECTORY.replace('X', str(sensor_num))
                
                # Get starting file number
                starting_file = self.get_starting_file_number(directory_path, '.csv')
                if starting_file is None:
                    input("\nPress Enter to continue...")
                    continue

                script_path = self.src_dir / "viz" / "plot_grid.py"
                self.run_script(script_path, args=[str(sensor_num), str(starting_file)])
                input("\nPress Enter to continue...")
                
            elif choice == "4":
                # Plot point cloud
                print("\n" + "="*40)
                print("POINT CLOUD VISUALIZATION")
                print("="*40)
                
                # Get sensor number
                sensor_num = self.get_sensor_number()
                if sensor_num is None:
                    input("\nPress Enter to continue...")
                    continue
                
                # Get directory path
                directory_path = LIDAR_DIRECTORY.replace('X', str(sensor_num))
                
                # Get starting file number
                starting_file = self.get_starting_file_number(directory_path, '.ply')
                if starting_file is None:
                    input("\nPress Enter to continue...")
                    continue
                
                script_path = self.src_dir / "viz" / "plot_pointcloud.py"
                self.run_script(script_path, args=[str(sensor_num), str(starting_file)])
                input("\nPress Enter to continue...")
                
            elif choice == "9":
                # Back to main menu
                break
                
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")

    def run(self):
        """Main application loop"""
        while True:
            self.clear_screen()
            self.print_header()
            self.print_main_menu()
            
            choice = self.get_user_input()
            
            if choice == "1":
                self.handle_data_generation()
                
            elif choice == "2":
                self.handle_preprocessing()
                
            elif choice == "3":
                self.handle_training()
                
            elif choice == "4":
                self.handle_visualization()
                
            elif choice == "9":
                print("\nExiting Framework Control Center...")
                break
                
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        cli = FrameworkCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
