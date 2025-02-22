from ffcv.writer import DatasetWriter
from ffcv.fields import *
import pickle
import torch
import os
import random
import math
import numpy as np
import gc
import sys
from concurrent.futures import ThreadPoolExecutor
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, Squeeze, ToDevice


if __name__ == "__main__":

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        #model = nn.DataParallel(model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        #model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Storing ID of current CUDA device
    #cuda_id = torch.cuda.current_device()
    #print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    #print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
    
    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *

    name = f"dataset{0}.beton"  # Define the path where the dataset will be written
    complete_path = os.path.join(FFCV_DIR, name)

    train_loader = Loader(complete_path, batch_size=16,
            num_workers=8, order=OrderOption.RANDOM,
            pipelines={
                'covariate': [NDArrayDecoder(), ToTensor(), ToDevice(torch.device('cuda:0'))],
                'label': [NDArrayDecoder(), ToTensor(), Squeeze(), ToDevice(torch.device('cuda:0'))]
            })
    print("finish")
    print(len(train_loader))
    #for batch in train_loader:
        #print(batch)
        #break  # Exit after printing the first batch

