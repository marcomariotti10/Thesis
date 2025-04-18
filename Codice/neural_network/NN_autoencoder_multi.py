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

def define_models(model_type, activation_function, model_name):

    model = model_type(activation_fn=activation_function)
    
    # Model creation
    model_path = os.path.join(MODEL_DIR, model_name + ".pth")
    checkpoint = torch.load(model_path, weights_only=True)

    # Remove 'module.' prefix from the state dict keys if it's there
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Create a new state dict without the "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Now load the cleaned state dict into your model
    model.load_state_dict(new_state_dict)
    
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.decoder.parameters():
        param.requires_grad = True

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

    #if "UNet" not in model_type.__name__:
        #calculate_dead_neuron_multi(model, device)

    model.decoder.apply(initialize_weights)

    print("----------------------------------------------------------------------")

    #if "UNet" not in model_type.__name__:
        #calculate_dead_neuron_multi(model, device)
    
    return model, device

def train(model, device, activation_function):
    
    optimizers = [
        torch.optim.Adam(decoder.parameters(), lr=1e-4)
        for decoder in model.decoder
    ]

    # Create one scheduler per head
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3)
        for opt in optimizers
    ]

    # Create one EarlyStopping per head
    early_stoppers = [
        EarlyStopping(patience=5, min_delta=0.0001)
        for _ in model.decoder
    ]
        
    pos_weight = torch.tensor([1], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)
   
    # Parameters for training
    early_stopping_triggered = False
    number_of_chuncks = NUMBER_OF_CHUNCKS
    num_total_epochs = 100
    num_epochs_for_each_chunck = 1
    number_of_chuncks_val = NUMBER_OF_CHUNCKS_TEST
    batch_size = BATCH_SIZE

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_val_losses = [float('inf')] * len(model.decoder)
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pth')

    for j in range(num_total_epochs):
        if early_stopping_triggered:
            break

        random.seed(SEED + j)
        train_order = random.sample(range(number_of_chuncks), number_of_chuncks)
        print("\nOrder of the chuncks for training:", train_order)
        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")

        for i in range(1):  # type: ignore
            if early_stopping_triggered:
                break

            print(f"\nChunck number {i+1} of {number_of_chuncks}")
            train_index = train_order[i]
            train_loader = load_dataset('train', train_index, device, batch_size)
            print("\nLenght of the train dataset:", len(train_loader))

            for epoch in range(num_epochs_for_each_chunck):
                start = datetime.now()
                model.train()
                head_train_losses = [0.0 for _ in model.decoder]
                for data in train_loader:
                    
                    inputs, targets = data
                    targets = [targets[:, i].unsqueeze(1).float().to(device) for i in range(len(FUTURE_TARGET_RILEVATION))]  # targets list: [target0, target1, target2, target3]
                    # targets list: [target0, target1, target2, target3]

                    latent = model.encoder(inputs)  # Forward pass through frozen encoder

                    # Forward pass through all heads
                    outputs = [decoder(latent) for decoder in model.decoder]

                    # Update each head separately
                    for head_idx in range(len(FUTURE_TARGET_RILEVATION)):
                        optimizers[head_idx].zero_grad()

                        loss = criterion(outputs[head_idx], targets[head_idx])
                        loss.backward()

                        optimizers[head_idx].step()
                        head_train_losses[head_idx] += loss.item()

                # Average the loss per head
                head_train_losses = [loss / len(train_loader) / number_of_chuncks for loss in head_train_losses]

                if epoch + 1 == num_epochs_for_each_chunck:
                    head_val_losses = [0.0 for _ in model.decoder]

                    for k in range(number_of_chuncks_val):
                        val_loader = load_dataset('val', k, device, batch_size)
                        model.eval()

                        with torch.no_grad():
                            for data in val_loader:
                                inputs, targets = data
                                targets = [
                                    targets[:, i].unsqueeze(1).float().to(device)
                                    for i in range(len(FUTURE_TARGET_RILEVATION))
                                ]

                                latent = model.encoder(inputs)
                                outputs = [decoder(latent) for decoder in model.decoder]

                                for z in range(len(FUTURE_TARGET_RILEVATION)):
                                    loss = criterion(outputs[z], targets[z])
                                    head_val_losses[z] += loss.item()

                    # Average the loss per head
                    head_val_losses = [loss / len(val_loader) / number_of_chuncks_val for loss in head_val_losses]

                    # Print and handle early stopping / schedulers
                    for head_idx, (val_loss, train_loss) in enumerate(zip(head_val_losses, head_train_losses)):
                        print(f"Head {head_idx + 1}     Train Loss: {train_loss:.4f}   Val Loss: {val_loss:.4f}")

                        if j >= 1 and val_loss < best_val_losses[head_idx]:
                            best_val_losses[head_idx] = val_loss
                            best_model_path = os.path.join(MODEL_DIR, f'best_model_head_{head_idx + 1}.pth')
                            torch.save(model.state_dict(), best_model_path)
                            print(f"New best model saved for Head {head_idx + 1} with Val Loss: {val_loss:.4f}")
                        
                        schedulers[head_idx].step(val_loss)
                        early_stoppers[head_idx](val_loss)
                    print("Epoch time:", datetime.now() - start)

                    if j >= 1 and all(val_loss < best for val_loss, best in zip(head_val_losses, best_val_losses)):
                        best_val_losses = head_val_losses.copy()
                        torch.save(model.state_dict(), best_model_path)
                        print(f"New best model saved with Val Losses: {best_val_losses}")

                    # Global stopping if all heads agree
                    if all(stopper.early_stop for stopper in early_stoppers):
                        print("Early stopping triggered for all heads.")
                        early_stopping_triggered = True
    
    #if "UNet" not in model_type.__name__:
        #calculate_dead_neuron_multi(model, device)
    
    del model

def test(model_type, device, model_name):

    batch_size = BATCH_SIZE

    print("\nLoading the best encoder and decoders...\n")

    model = model_type(activation_fn=activation_function)

    '''

    model_path = os.path.join(MODEL_DIR, 'best_model_head_1.pth')

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

    '''
    # Load encoder
    best_encoder_path = os.path.join(MODEL_DIR, model_name + '.pth')
    encoder_state_dict = torch.load(best_encoder_path, map_location=device, weights_only=True)

    # Remove "encoder." prefix from the keys
    encoder_state_dict = {k.replace("encoder.", ""): v for k, v in encoder_state_dict.items() if k.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state_dict, strict=True)
    print(f"Encoder loaded from {best_encoder_path}")

    # Load each decoder head
    for idx in range(4):
        best_head_path = os.path.join(MODEL_DIR, f'best_model_head_{idx+1}.pth')
        full_state_dict = torch.load(best_head_path, map_location=device, weights_only=True)

        # Extract only the decoder parameters for the correct head
        head_state_dict = {}
        for name, param in full_state_dict.items():
            if name.startswith(f"decoder.{idx}"):
                #print("name", name)
                clean_name = name.replace(f"decoder.{idx}.", "")
                #print(f"Clean name: {clean_name}")
                head_state_dict[clean_name] = param

        model.decoder[idx].load_state_dict(head_state_dict, strict=True)
        print(f"Head {idx+1} loaded from {best_head_path}")
    
    #'''

    model.to(device)
    model.eval()  # Set the model to evaluation mode
    print("\nAll components loaded successfully!\n")

    model = model.to(device)

    # Check for multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))
    else:
        summary(model, input_size=(NUMBER_RILEVATIONS_INPUT, 400, 400))

    pos_weight = torch.tensor([1], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    test_losses = []
    for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")
        test_loader = load_dataset('test', i, device, batch_size)
        print("\nLenght test dataset: ", len(test_loader))
        gc.collect()

        model.eval()
        head_test_losses = [0.0 for _ in model.decoder]

        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = [
                    targets[:, i].unsqueeze(1).float().to(device)
                    for i in range(len(FUTURE_TARGET_RILEVATION))
                ]

                latent = model.encoder(inputs)
                outputs = [decoder(latent) for decoder in model.decoder]

                for z in range(len(FUTURE_TARGET_RILEVATION)):
                    loss = criterion(outputs[z], targets[z])
                    head_test_losses[z] += loss.item()

    # Average the loss per head
    head_test_losses = [loss / len(test_loader) / NUMBER_OF_CHUNCKS_TEST for loss in head_test_losses]

    # Print and handle early stopping / schedulers
    for head_idx, test_loss in enumerate(head_test_losses):
        print(f"Head {head_idx + 1}     Test Loss: {test_loss:.4f}")
        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)
    return total_loss, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and test a neural network model.')
    parser.add_argument('--model_type', type=str, default='MultiHeadAutoencoder', help='Type of model to use')
    parser.add_argument('--activation_function', type=str, default='ReLU', help='Activation function to apply to the model')
    parser.add_argument('--encoder', type=str, default='model_ENCODER_20250415_124611_loss_0.0040_MultiHeadAutoencoder', help='Pre-trained encoder to use')
    args = parser.parse_args()

    model_type = globals()[args.model_type]
    activation_function = getattr(nn, args.activation_function)
    model_name = args.encoder

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

    model, device = define_models(model_type, activation_function, model_name)

    try:
    
        train(model, device, activation_function)

        total_loss, model_best = test(model_type, device, model_name)

        # Define the directory where you want to save the model
        os.makedirs(MODEL_DIR, exist_ok=True)
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f'model_{time}_loss_{total_loss:.4f}_{model_type.__name__}.pth'
        model_save_path = os.path.join(MODEL_DIR, model_name)
        torch.save(model_best.state_dict(), model_save_path)
        print(f'Model saved : {model_name}')

    except KeyboardInterrupt:
        print("Training interrupted by user.")

        torch.distributed.destroy_process_group()