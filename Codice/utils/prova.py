import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv
import os
import numpy as np    
import argparse
import cProfile
import pstats
import sys
from sklearn.model_selection import train_test_split
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from torchsummary import summary
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
import gc
from multiprocessing import Pool, set_start_method
from matplotlib.path import Path
import open3d as o3d
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import pickle
from datetime import datetime
import random
import math
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

# Create a dummy dataset (replace with real dataloader)
from torch.utils.data import DataLoader, TensorDataset

N = 100  # Number of samples
past_frames = torch.rand(N, 5, 400, 400, dtype = torch.float16)  # (N, 5, 400, 400)
future_frames = torch.randint(0, 1, (N, 1, 400, 400), dtype=torch.float16)  # (N, 1, 400, 400)

dataset = TensorDataset(past_frames, future_frames)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Initialize and train model
model = ConditionalUNet()

train_model(model, dataloader, epochs=3)

# Inference example
sample_past_frames = past_frames[:1]  # Take one sample
predicted_future_frame = sample_future_frame(model, sample_past_frames)