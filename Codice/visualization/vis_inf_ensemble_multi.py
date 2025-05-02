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
    
def visualize_prediction(pred, model_pred, gt, map):
    """
    Visualize the grid map and the prediction.
    
    Parameters:
    - grid_map: numpy array of shape (400, 400)
    - prediction: numpy array of shape (400, 400)
    """

    for i in range(len(FUTURE_TARGET_RILEVATION)):
        fig, ax = plt.subplots(2, len(model_pred), figsize=(10, 10))
        ax[0, 0].imshow(map, cmap='gray', alpha=0.5)
        ax[0, 0].imshow(pred[i], cmap='jet', alpha=0.5)
        ax[0, 0].set_title('Overlay of Original and Ensemble prediction Grid Maps')

        ax[0, 1].imshow(map, cmap='gray', alpha=0.5)
        ax[0, 1].imshow(gt[i], cmap='jet', alpha=0.5)
        ax[0, 1].set_title('Overlay of Original and Ground Truth Grid Maps')

        for k in range(len(model_pred)):
            ax[1, k].imshow(map, cmap='gray', alpha=0.5)
            ax[1, k].imshow(model_pred[k][i], cmap='jet', alpha=0.5)
            ax[1, k].set_title(f'Overlay of Original and Prediction Grid Maps {k+1}')

        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized()  # For Qt5Agg backend (common in Anaconda, Linux, etc.)
        except AttributeError:
            # If neither works, just set a large size
            fig.set_size_inches(18.5, 10.5)
        plt.show()
        plt.show()

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


def evaluate(models, device, ensemble_name):
    
    test_losses = [[0] * 5 for _ in range(len(models) + 1)]
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the saved ensemble model
    model_save_path = os.path.join(MODEL_DIR, ensemble_name + '.pth')
    ensemble = EnsembleModel(len(models))  # Initialize the model
    ensemble.load_state_dict(torch.load(model_save_path, weights_only = True))  # Load the saved state dict
    ensemble.to(device)  # Move to the appropriate device

    print("Ensemble model weights:")
    print(ensemble.weights.data)

    ensemble.eval()

    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"\nTest chunk number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('test', i, device, BATCH_SIZE)
        print("\nLength test dataset: ", len(test_loader))

        test_losses_chunk = [[0] * 5 for _ in range(len(models) + 1)]  # 4 heads + average loss
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = [
                    targets[:, 0].unsqueeze(1).float(),  # Future map at time 1
                    targets[:, 1].unsqueeze(1).float(),  # Future map at time 2
                    targets[:, 2].unsqueeze(1).float(),  # Future map at time 3
                    targets[:, 3].unsqueeze(1).float(),  # Future map at time 4
                ]

                # Get outputs from all models
                model_outputs = [torch.stack(model(inputs), dim=1) for model in models]  # Each output: (B, 4, H, W)

                # Pass stacked outputs through the ensemble model
                combined_outputs = ensemble(model_outputs)  # Shape: (B, 4, H, W)

                # Compute losses for each model
                for model_idx, output in enumerate(model_outputs):
                    for head_idx in range(4):  # Iterate over the 4 heads
                        loss = criterion(output[:, head_idx], targets[head_idx]).item()
                        test_losses_chunk[model_idx][head_idx] += loss
                

                # Compute losses for the ensemble model
                for head_idx in range(4):  # Iterate over the 4 heads
                    loss = criterion(combined_outputs[:, head_idx], targets[head_idx]).item()
                    test_losses_chunk[-1][head_idx] += loss

        for model_idx in range(len(models)):
            #print(sum(test_losses_chunk[model_idx][:4]))
            avg_loss = sum(test_losses_chunk[model_idx][:4]) / 4
            test_losses_chunk[model_idx][4] += avg_loss

        avg_loss = sum(test_losses_chunk[-1][:4]) / 4
        test_losses_chunk[-1][4] += avg_loss

        # Normalize losses by the number of batches
        test_losses_chunk = [[loss / len(test_loader) for loss in losses] for losses in test_losses_chunk]

        # Accumulate test_losses_chunk into test_losses
        for model_idx in range(len(test_losses)):
            for head_idx in range(5):  # Include the average loss
                test_losses[model_idx][head_idx] += test_losses_chunk[model_idx][head_idx]

        # Print losses for each model
        for model_idx, losses in enumerate(test_losses_chunk[:-1]):
            print(f"\nModel {model_idx + 1} losses:")
            for head_idx, loss in enumerate(losses[:4]):
                print(f"  Head {head_idx + 1} Loss: {loss:.4f}")
            print(f"  Average Loss: {losses[4]:.4f}")

        # Print losses for the ensemble model
        print("\nEnsemble model losses:")
        for head_idx, loss in enumerate(test_losses_chunk[-1][:4]):
            print(f"  Head {head_idx + 1} Loss: {loss:.4f}")
        print(f"  Average Loss: {test_losses_chunk[-1][4]:.4f}")

    # Compute the average losses across all chunks
    test_losses = [[loss / NUMBER_OF_CHUNCKS_TEST for loss in losses] for losses in test_losses]

    print("\n----------------------------Total losses----------------------------")

    # Print losses for each model
    for model_idx, losses in enumerate(test_losses[:-1]):
        print(f"\nModel {model_idx + 1} total losses:")
        for head_idx, loss in enumerate(losses[:4]):
            print(f"  Head {head_idx + 1} Loss: {loss:.4f}")
        print(f"  Average Loss: {losses[4]:.4f}")

    # Print losses for the ensemble model
    print("\nEnsemble model total losses:")
    for head_idx, loss in enumerate(test_losses[-1][:4]):
        print(f"  Head {head_idx + 1} Loss: {loss:.4f}")
    print(f"  Average Loss: {test_losses[-1][4]:.4f}")

def show_predictions(models, device, ensemble_name):

    # Load the saved ensemble model
    model_save_path = os.path.join(MODEL_DIR, ensemble_name + '.pth')
    ensemble = EnsembleModel(len(models))  # Initialize the model
    ensemble.load_state_dict(torch.load(model_save_path, weights_only = True))  # Load the saved state dict
    ensemble.to(device)  # Move to the appropriate device

    ensemble.eval()

    sigmoid = torch.nn.Sigmoid()

    print("\n-----------------------------Show predictions----------------------------")

    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"\nChunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}")

        test_loader = load_dataset('test', i, device, 1)
        print("\nLenght of the datasets:", len(test_loader))

        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                predictions = []
                model_predictions = []
                grid_maps = []
                vertices = []
                inputs, targets = data
                targets = [
                    targets[:, i].unsqueeze(1).float().to(device)
                    for i in range(len(FUTURE_TARGET_RILEVATION))
                ]

                model_outputs = [torch.stack(model(inputs), dim=1) for model in models]

                print(model_outputs[0].shape)

                x_t = ensemble(model_outputs)  # Shape: (B, 4, H, W)

                x_t = torch.cat([x_t], dim=1)
                print(x_t.shape)


                # Apply threshold to convert outputs to 0 or 1
                threshold = 0.4
                x_t = (x_t > threshold).float()

                # Calculate loss with true noise
                x_t = sigmoid(x_t)
                predictions.append(x_t)
                predictions = torch.cat(predictions).cpu().numpy()
                #print("Predictions Shape:", predictions.shape)

                for z in range(len(models)):
                    
                    # Apply threshold to convert outputs to 0 or 1
                    threshold = 0.4
                    model_pred = (model_outputs[z] > threshold).float()

                    # Calculate loss with true noise
                    model_pred = sigmoid(model_pred)
                    model_predictions.append(model_pred)
               
                model_predictions = torch.cat(model_predictions).cpu().numpy()
                #print("Predictions Shape:", predictions.shape)
                
                grid_maps = data[0].cpu().numpy()
                vertices = data[1].cpu().numpy()

                # Now covariate_data and label_data are numpy arrays containing all the elements
                #print("Covariate data shape:", grid_maps.shape)
                #print("Label data shape:", vertices.shape)

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                vertices = vertices.reshape(-1, 400, 400)
                model_predictions = [pred.reshape(-1, 400, 400) for pred in model_predictions]
                
                print(predictions.shape)
                print(model_predictions[0].shape)
                visualize_prediction(predictions, model_predictions, vertices, grid_maps[i])


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

    evaluate(models, device, ensemble_name)

    try:

        show_predictions(models, device, ensemble_name)

    except KeyboardInterrupt:
        
        print("Program interrupted by user.")

    dist.destroy_process_group()
