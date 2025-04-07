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
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, Squeeze, ToDevice, ToTorchImage, RandomHorizontalFlip
from sklearn.preprocessing import MinMaxScaler
from ffcv.reader import Reader
import torch.nn.functional as F
import torch.distributed as dist

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

def define_models(model_type, activation_function):
    # Model creation
    model = model_type(activation_fn=activation_function)
    model.apply(initialize_weights)

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

    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))
    else:
        summary(model, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))
    
    return model, device

def train(model, device, activation_function):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    early_stopping = EarlyStopping(patience=5, min_delta=0.0001)
    
    pos_weight = torch.tensor([1], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
   
    # Parameters for training
    early_stopping_triggered = False
    number_of_chuncks = NUMBER_OF_CHUNCKS
    num_total_epochs = 100
    num_epochs_for_each_chunck = 1
    number_of_chuncks_val = NUMBER_OF_CHUNCKS_TEST
    batch_size = 8

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_loss = float('inf')  # Initialize best validation loss
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')

    alpha_cumprod = get_noise_schedule()

    for j in range(num_total_epochs):
        if early_stopping_triggered:
            break

        random.seed(SEED + j)
        train_order = random.sample(range(number_of_chuncks), number_of_chuncks)
        print("\nOrder of the chuncks for training:", train_order)
        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")

        for i in range(number_of_chuncks):  # type: ignore
            if early_stopping_triggered:
                break

            print(f"\nChunck number {i+1} of {number_of_chuncks}")
            train_index = train_order[i]
            train_loader = load_dataset('train', train_index, device, batch_size)
            print("\nLenght of the train dataset:", len(train_loader))

            for epoch in range(num_epochs_for_each_chunck):
                start = datetime.now()
                model.train()
                train_loss = 0
                for data in train_loader:
                    
                    inputs, targets = data
                    targets = targets.float()

                    optimizer.zero_grad()

                    # Predict the noise for this timestep
                    pred = model(inputs)

                    # Calculate loss with true noise
                    loss = criterion(pred, targets)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                if epoch + 1 == num_epochs_for_each_chunck:
                    val_losses = []
                    for k in range(number_of_chuncks_val):
                        val_loader = load_dataset('val', k, device, batch_size)
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for data in val_loader:
                                inputs, targets = data
                                targets = targets.float()

                                # Predict the noise for this timestep
                                pred = model(inputs)

                                # Calculate loss with true noise
                                loss = criterion(pred, targets)
            
                                val_loss += loss.item()
                        val_loss /= len(val_loader)
                        val_losses.append(val_loss)

                    total_val_loss = sum(val_losses) / len(val_losses)
                    print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}, Val Loss: {total_val_loss:.4f}        Time: {datetime.now() - start}')

                    if j >= 1:
                        if total_val_loss < best_val_loss:
                            best_val_loss = total_val_loss
                            torch.save(model.state_dict(), best_model_path)
                            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

                        scheduler.step(val_loss)
                        early_stopping(val_loss)
                        if early_stopping.early_stop:
                            print("Early stopping triggered")
                            early_stopping_triggered = True
                            break
                else:
                    print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}                          Time: {datetime.now() - start}')

    # Get the first batch of data
    loader = load_dataset('val', 0, device, batch_size)
    first_batch = next(iter(loader))

    # Unpack the inputs and targets from the first batch
    inputs, targets = first_batch
    check_dead_neurons_autoencoder(model, inputs, targets, activation_function)
    
    del model

def test(model_type, device):

    batch_size = 8

    model = model_type(activation_fn=activation_function)

    model_path = os.path.join(MODEL_DIR, 'best_model.pth')

    checkpoint = torch.load(model_path, map_location=device)

    # Remove 'module.' prefix from the state dict keys if it's there
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Create a new state dict without the "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Now load the cleaned state dict into your model
    model.load_state_dict(new_state_dict)

    model = model.to(device)

    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))
    else:
        summary(model, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))

    alpha_cumprod = get_noise_schedule()

    pos_weight = torch.tensor([1], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    test_losses = []
    for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")
        test_loader = load_dataset('test', i, device, batch_size)
        print("\nLenght test dataset: ", len(test_loader))
        gc.collect()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()
                
                # Predict the noise for this timestep
                pred = model(inputs)

                # Calculate loss with true noise
                loss = criterion(pred, targets)

                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)
    return total_loss, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and test a neural network model.')
    parser.add_argument('--model_type', type=str, default='Autoencoder_big_big', help='Type of model to use')
    parser.add_argument('--activation_function', type=str, default='ReLU', help='Activation function to apply to the model')
    args = parser.parse_args()

    model_type = globals()[args.model_type]
    activation_function = getattr(nn, args.activation_function)

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

    model, device = define_models(model_type, activation_function)
    
    train(model, device, activation_function)

    total_loss, model_best = test(model_type, device)

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{total_loss:.4f}_{model_type.__name__}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model_best.state_dict(), model_save_path)
    print(f'Model saved : {model_name}')