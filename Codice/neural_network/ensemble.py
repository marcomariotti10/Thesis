import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import numpy as np
import os
import torch.distributed as dist
from torch import nn
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from torch.utils.data import DataLoader, TensorDataset

class EnsembleModel(nn.Module):
    def __init__(self, num_models=2, num_outputs=4):
        super(EnsembleModel, self).__init__()
        self.num_models = num_models
        self.num_outputs = num_outputs
        
        # Learnable weights: (num_models, num_outputs, 1, 1, 1)
        self.weights = nn.Parameter(torch.ones(num_models, num_outputs, 1, 1, 1))
    
    def forward(self, outputs):
        """
        outputs: list of tensors, each shape [batch_size, 4, 1, 400, 400]
        Example: [model1_out, model2_out]
        """
        stacked = torch.stack(outputs, dim=0)  # shape [num_models, batch, 4, 1, 400, 400]
        softmax_weights = torch.softmax(self.weights, dim=0)  # shape [num_models, 4, 1, 1, 1]
    
        weighted = stacked * softmax_weights.unsqueeze(1)  # broadcast weights over batch
        combined = weighted.sum(dim=0)  # sum over models → shape [batch, 4, 1, 400, 400]
        
        return combined

def model_preparation(model_names, models_types):
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    models = []

    for model_name in model_names:
        model_path = os.path.join(MODEL_DIR, model_name + '.pth')
        model = models_types[model_names.index(model_name)]

        checkpoint = torch.load(model_path, map_location=device, weights_only = True)

        # Remove 'module.' prefix from the state dict keys if it's there
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Create a new state dict without the "module." prefix
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = value

        # Now load the cleaned state dict into your model
        model.load_state_dict(new_state_dict)
        model.eval()
        models.append(model)
        model = model.to(device)

        # Check for multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
            model = nn.DataParallel(model)

    return models, device

def train_ensemble(models, device, models_types):

    ensemble = EnsembleModel(len(models)).to(device)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()  # or MSELoss if smoother grayscale output
    number_of_epochs = 1

    for j in range(number_of_epochs):

        print(f"\nTraining ensemble model {j+1} of {number_of_epochs}: ")

        for i in range(NUMBER_OF_CHUNCKS_TEST):
            print(f"\nValidation chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

            loader = load_dataset('val', i, device, BATCH_SIZE)
            print("\nLenght val dataset: ", len(loader))

            train_loss = 0

            for inputs, targets in loader:
                #torch.cuda.empty_cache()
                targets = [
                            targets[:, 0].unsqueeze(1).float(),  # Future map at time 1
                            targets[:, 1].unsqueeze(1).float(),  # Future map at time 2
                            targets[:, 2].unsqueeze(1).float(),  # Future map at time 3
                            targets[:, 3].unsqueeze(1).float(),  # Future map at time 4
                        ]

                optimizer.zero_grad()
                
                # Get outputs from all models
                model_outputs = [torch.stack(model(inputs), dim=1) for model in models]  # Each output: (B, 4, 1, H, W)

                #print(stacked_outputs.shape)
                # Pass stacked outputs through the ensemble model
                combined_outputs = ensemble(model_outputs)  # Shape: (B, 4, H, W)

                # Compute the loss
                loss = 0
                #print(combined_outputs.shape)
                for z in range(4):  # Iterate over the 4 outputs
                    loss += criterion(combined_outputs[:, z], targets[z])  # Ensure shapes match
                loss /= 4  # Average over all outputs

                # Backpropagation
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(loader)
            print(f'Train Loss: {train_loss:.4f}')

    print("\n----------------------------Test ensemble model----------------------------")

    test_losses = []
    for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")
        test_loader = load_dataset('test', i, device, BATCH_SIZE)
        print("\nLenght test dataset: ", len(test_loader))
        gc.collect()

        ensemble.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data

                targets = [
                            targets[:, 0].unsqueeze(1).float(),  # Future map at time 1
                            targets[:, 1].unsqueeze(1).float(),  # Future map at time 2
                            targets[:, 2].unsqueeze(1).float(),  # Future map at time 3
                            targets[:, 3].unsqueeze(1).float(),  # Future map at time 4
                        ]
                
                # Get outputs from all models
                model_outputs = [torch.stack(model(inputs), dim=1) for model in models]  # Each output: (B, 4, H, W)

                #print(stacked_outputs.shape)
                # Pass stacked outputs through the ensemble model
                combined_outputs = ensemble(model_outputs)  # Shape: (B, 4, H, W)

                # Compute the loss
                loss = 0
                print(combined_outputs.shape)
                for z in range(4):  # Iterate over the 4 outputs
                    loss += criterion(combined_outputs[:, z], targets[z])  # Ensure shapes match
                loss /= 4  # Average over all outputs

                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')
        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)


    os.makedirs(MODEL_DIR, exist_ok=True)
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_identifiers = "_".join([type(model).__name__ for model in models_types])  # Use class names
    model_name = f'Ensemble_model_{time}_loss_{total_loss:.4f}_{model_identifiers}.pth'    
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(ensemble.state_dict(), model_save_path)
    print(f'Model saved : {model_name}')


def find_best_threshold(models, device):
    test_losses = [[] for _ in range(len(models) + 1)]
    criterion = torch.nn.BCELoss()
    sigmoid = torch.nn.Sigmoid()
    thresholds = np.arange(0.1, 1.0, 0.1)

    chunk_best_thresholds = []
    chunk_best_losses = []

    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"\nTest chunk number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('test', i, device, 16)
        print("\nLength test dataset: ", len(test_loader))

        best_threshold = 0.5
        best_loss = float('inf')

        with torch.no_grad():
            for threshold in thresholds:
                total_loss = 0
                for inputs, targets in test_loader:
                    targets = targets.float()
                    outputs = [model(inputs) for model in models]
                    avg_output = sum(outputs) / len(outputs)
                    avg_output = sigmoid(avg_output)
                    avg_output = (avg_output > threshold).float()
                    loss = criterion(avg_output, targets)
                    total_loss += loss.item()
                avg_loss = total_loss / len(test_loader)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_threshold = threshold

        chunk_best_thresholds.append(best_threshold)
        chunk_best_losses.append(best_loss)
        print(f"Best threshold for chunk {i+1}: {best_threshold}, Loss: {best_loss:.4f}")

    # Find the general best threshold among all chunks
    general_best_threshold = chunk_best_thresholds[np.argmin(chunk_best_losses)]

    print(f"\nGeneral best threshold: {general_best_threshold}")

if __name__ == '__main__':

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # This is the address of the master node
    os.environ['MASTER_PORT'] = '29500'     # This is the port for communication (can choose any available port)

    # Set other environment variables for single-node multi-GPU setup
    os.environ['RANK'] = '0'       # Process rank (0 for single process)
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes
    os.environ['LOCAL_RANK'] = '0'  # Local rank for single-GPU (0 for single GPU)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')  # Use NCCL for multi-GPU setups
    
    model_names = [
        'model_ENCODER_20250501_114340_loss_0.0221_MultiHeadUNetAutoencoder',
        'model_ENCODER_20250501_115617_loss_0.0217_MultiHeadCBAMAutoencoder'
    ]

    ensemble_name = "Ensemble_model_20250502_123438_loss_0.0209_MultiHeadUNetAutoencoder_MultiHeadCBAMAutoencoder"

    models_types = [
        MultiHeadUNetAutoencoder(),
        MultiHeadCBAMAutoencoder()
    ]

    models, device = model_preparation(model_names, models_types)

    #find_best_threshold(models, device)

    train_ensemble(models, device, models_types)

    dist.destroy_process_group()
