import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from neural_network import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import csv
import os
import numpy as np
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
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, Squeeze, ToDevice, ToTorchImage, RandomHorizontalFlip
from sklearn.preprocessing import MinMaxScaler
from ffcv.reader import Reader
import torch.nn.functional as F
import torch.distributed as dist

def check_dead_neurons(model, input_data):
    model.eval()
    with torch.no_grad():
        activations = model(input_data)
        num_zeros = (activations == 0).sum().item()
        total_neurons = activations.numel()
        zero_percentage = (num_zeros / total_neurons) * 100
        print(f"Dead neurons: {zero_percentage:.2f}%")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Avoids division by zero

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)  # Ensure values are between 0 and 1
        numerator = 2 * (preds * targets).sum() + self.smooth
        denominator = preds.sum() + targets.sum() + self.smooth
        return 1 - (numerator / denominator)  # 1 - Dice Score
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): 
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha, list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        # Ensure input and target have shape (1, 400, 400)
        input = input.view(1, -1)  # Flatten input to shape (1, H*W)
        target = target.view(1, -1).long()  # Flatten target to shape (1, H*W) and convert to int64
        
        logpt = F.log_softmax(input, dim=1)  # Apply log softmax along the last dimension
        logpt = logpt.gather(1, target)  # Get the log probability of the target class
        logpt = logpt.view(-1)  # Flatten to 1D

        pt = logpt.exp()  # Convert log probability to probability (p_t)

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))  # Select alpha for each class based on the target class
            logpt = logpt * at  # Apply alpha weighting

        loss = -1 * (1 - pt)**self.gamma * logpt  # Compute focal loss

        if self.size_average:
            return loss.mean()  # Return average loss
        else:
            return loss.sum()  # Return total loss
        
class WeightedBCELoss(nn.Module):
    def __init__(self, weight_background=1.0, weight_object=100.0):
        super(WeightedBCELoss, self).__init__()
        # Weights for the background (0) and object (1) classes
        self.weight_background = weight_background
        self.weight_object = weight_object

    def forward(self, predictions, targets):
        # Predictions should be a probability (output of sigmoid), and targets should be 0 or 1
        # Apply sigmoid to the predictions to convert them to probabilities
        predictions = predictions.sigmoid()
        
        # Binary Cross-Entropy calculation
        bce_loss = - (self.weight_object * targets * torch.log(predictions + 1e-8) + 
                      self.weight_background * (1 - targets) * torch.log(1 - predictions + 1e-8))
        
        # Return the mean of the loss
        return bce_loss.mean()

def load_dataset(name,i,device):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_name_chunck_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    
    complete_path_train = os.path.join(complete_name_chunck_path, name_train)

    train_loader = Loader(complete_path_train, batch_size=16,
    num_workers=8, order=OrderOption.RANDOM, distributed = True, seed = SEED, drop_last=True,
    os_cache=False,
    pipelines={
        'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                    #RandomHorizontalFlip(0.3),
                    ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                    ToDevice(device, non_blocking=True)],
        'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                #RandomHorizontalFlip(0.3),
                ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                ToDevice(device, non_blocking=True)]
    })

    return train_loader

if __name__ == "__main__":

    
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # This is the address of the master node
    os.environ['MASTER_PORT'] = '29500'     # This is the port for communication (can choose any available port)

    # Set other environment variables for single-node multi-GPU setup
    os.environ['RANK'] = '0'       # Process rank (0 for single process)
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes
    os.environ['LOCAL_RANK'] = '0'  # Local rank for single-GPU (0 for single GPU)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')  # Use NCCL for multi-GPU setups
    
    gc.collect()

    random.seed(SEED)

    # Model creation
    model = Autoencoder_classic()
    model.apply(weights_init)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    pos_weight = torch.tensor([10]).to(device)  # Must be a tensor!
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if isinstance(model, nn.DataParallel):
        summary(model.module, (1, 400, 400))
    else:
        summary(model, (1, 400, 400))

    # Parameters for training
    early_stopping_triggered = False
    number_of_chucks= NUMBER_OF_CHUNCKS
    num_total_epochs = 50
    num_epochs_for_each_chunck = 3
    number_of_chucks_testset = NUMBER_OF_CHUNCKS_TEST

    best_val_loss = float('inf')  # Initialize best validation loss
    best_model_path = os.path.join(MODEL_DIR, "best_model.pth")  # Path to save the best model

    for j in range(num_total_epochs):
        
        if early_stopping_triggered:
            break

        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")
        random.seed(SEED + j)

        for i in range(number_of_chucks): #type: ignore
            
            if early_stopping_triggered:
                break
            
            print(f"\nChunck number {i+1} of {number_of_chucks}")

            with ThreadPoolExecutor(max_workers=2) as executor:
                train_loader, val_loader = executor.map(load_dataset, ['train', 'val'], [i, i], [device, device])

            print("\nLenght of the datasets:", len(train_loader), len(val_loader))


            for epoch in range(num_epochs_for_each_chunck):
                
                start = datetime.now()
                model.train()
                train_loss = 0
                for data in train_loader:
                    inputs, targets = data
                    targets = targets.float()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        inputs, targets = data
                        targets = targets.float()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                scheduler.step(val_loss)
                print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')  

                if j >= 2:
                   
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        #torch.save(model.state_dict(), best_model_path)
                        #print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        early_stopping_triggered = True
                        break
                else:
                    print("too early to stop")
                
                print(f"Time to move data to GPU: {datetime.now() - start}")

    # Get the first batch of data
    first_batch = next(iter(train_loader))

    # Unpack the inputs and targets from the first batch
    inputs, targets = first_batch
    check_dead_neurons(model, inputs)

    print("\n-------------------------------------------")
    print("Training completed")
    print("-------------------------------------------\n") 

    gc.collect()

    test_losses = []
    i = 0

    for i in range(number_of_chucks_testset): #type: ignore

        print(f"\nTest chunck number {i+1} of {number_of_chucks_testset}: ")

        name_test = f"dataset_test{i}.beton"  # Define the path where the dataset will be written
        complete_path_train = os.path.join(FFCV_DIR, name_test)

        test_loader = load_dataset('test', i, device)

        print("\nLenght test dataset: ", len(test_loader))

        gc.collect()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{total_loss:.4f}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved : {model_name}')