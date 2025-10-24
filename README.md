# Thesis Project: V2I-Based Framework for Road User Future Position Prediction using LiDAR HeightMap

This repository contains the complete implementation of a thesis project focused on autonomous vehicle perception using LiDAR sensors and deep learning models. The project includes data generation through CARLA simulation, preprocessing pipelines, and neural network training for occupancy grid prediction.

## 🚗 Project Overview

This thesis project implements an end-to-end pipeline for autonomous vehicle perception using:
- **CARLA Simulator** for realistic driving scenario generation
- **Multi-LiDAR sensor arrays** for 360° environmental perception  
- **Occupancy grid representations** for spatial understanding
- **Deep learning models** (Autoencoders and Diffusion models) for future state prediction

## 📁 Project Structure

```
├── scripts/                   # Main execution scripts and configuration
│   ├── CLI.py                 # Command-line interface for the entire framework
│   ├── settings.py            # Global configuration parameters
│   └── sensors_specs/         # LiDAR sensor configuration files
│       ├── lidar1.json        # Individual sensor specifications
│       ├── lidar2.json        # (12 total sensors)
│       └── ...
│
└── src/                            # Source code modules
    ├── config/                     # Configuration and helper modules
    │   ├── directories.py          # Directory structure management
    │   ├── helpers.py              # Utility functions
    │   └── model_architectures.py  # Neural network architectures
    │
    ├── data_generation/       # CARLA simulation and data collection
    │   ├── simulation.py      # Main simulation controller
    │   ├── sensors.py         # LiDAR sensor implementation
    │   ├── snapshots.py       # Data snapshot management
    │   └── spawn_actors.py    # Vehicle and pedestrian spawning
    │
    ├── preprocessing/              # Data preprocessing pipeline
    │   ├── convert_pointcloud.py   # Point cloud to occupancy grid conversion
    │   ├── convert_snapshot.py     # Snapshot format conversion
    │   ├── filtering_snapshot.py   # Data filtering and cleaning
    │   ├── sync_data.py            # Multi-sensor data synchronization
    │   ├── augment_combine.py      # Data augmentation techniques
    │   ├── build_ffcv.py           # Fast data loading optimization
    │   ├── fit_scaler.py           # Data normalization
    │   └── shard_dataset.py        # Dataset partitioning
    │
    ├── models/                # Deep learning models
    │   ├── autoencoder.py     # Autoencoder implementation and training
    │   └── diffusion.py       # Diffusion model implementation and training
    │
    └── viz/                    # Visualization and analysis tools
        ├── plot_pointcloud.py  # Point cloud visualization
        ├── plot_grid.py        # Heightmap
        
         visualization
        ├── plot_autoencoder.py # Autoencoder results visualization
        └── plot_diffusion.py   # Diffusion model results visualization
```

## 🚀 Quick Start

### Prerequisites
```bash
# Core dependencies
pip install torch torchvision torchaudio

# Computer vision and scientific computing

pip install opencv-python numpy matplotlib scipy

# Machine learning and data processing
pip install scikit-learn pandas

# 3D visualization and processing
pip install open3d

# Model utilities and performance optimization
pip install torchsummary ffcv

# CARLA Python API (ensure CARLA simulator is installed)
pip install carla

# Install CARLA Simulator (0.9.11+)
# Download from: https://carla.org/
```

### Setup
1. **Configure Project Path**: Edit `HOME` variable in `scripts/settings.py`
2. **Verify CARLA Installation**: Ensure CARLA server is accessible
3. **Check Sensor Configurations**: Review `scripts/sensors_specs/` files

### Running the Framework
```bash
# Start the interactive framework
cd scripts/
python CLI.py

# Follow the menu prompts:
# 1. Generate Data → Set up simulation and collect sensor data
# 2. Preprocess Data → Convert and augment the collected data  
# 3. Train Neural Network → Train autoencoder or diffusion models
# 4. Visualization → Analyze results and generate plots
```

## 🔧 Advanced Usage

### Custom Sensor Configurations
Modify JSON files in `sensors_specs/` to adjust:
- Sensor positions and orientations
- LiDAR specifications (range, resolution, FOV)
- Data collection parameters

### Hyperparameter Tuning
Edit `settings.py` to customize:
- Grid resolution and dimensions
- Augmentation strategies and intensities
- Training parameters and model architectures
- Diffusion noise schedules

### Custom Models
Extend `src/models/` with new architectures:
- Implement in `model_architectures.py`
- Add training scripts following existing patterns
- Integrate with CLI framework

## 📊 Output and Results

The framework generates:
- **Raw Data**: Point clouds and sensor measurements
- **Processed Data**: Occupancy grids and synchronized datasets  
- **Trained Models**: Saved model checkpoints and configurations
- **Visualizations**: Plots, animations, and analysis reports


## 📄 License

This project is part of academic research. Please refer to institutional guidelines for usage and distribution.

---

**Author**: Marco Mariotti  
**Institution**: Polytechnic of Milan 
**Year**: 2025