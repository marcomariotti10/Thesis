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
    def __init__(self, lenght):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(lenght))  # Start with equal weights
        self.lenght = lenght

    def forward(self, outputs):  # outputs: (B, 3, H, W)
        norm_weights = F.softmax(self.weights, dim=0)  # shape: (3,)
        # Apply weights to each channel, then sum along channel dimension
        weighted_output = (outputs * norm_weights.view(1, self.lenght, 1, 1)).sum(dim=1)  # shape: (B, H, W)
        return weighted_output.unsqueeze(1)  # Add channel dimension → (B, 1, H, W)

def visualize_prediction(pred, gt, map, preds):
    num_preds = len(preds)
    fig, ax = plt.subplots(2, num_preds, figsize=(12, 16))

    ax[0, 0].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 0].imshow(pred[0], cmap='jet', alpha=0.5)
    ax[0, 0].set_title('Overlay of Original and Average prediction Grid Maps')

    ax[0, 1].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 1].imshow(pred[1], cmap='jet', alpha=0.5)
    ax[0, 1].set_title('Overlay of Original and Ensemble prediction Grid Maps')

    ax[0, 2].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 2].imshow(gt, cmap='jet', alpha=0.5)
    ax[0, 2].set_title('Overlay of Original and Ground Truth Grid Maps')

    for i in range(num_preds):
        ax[1, i].imshow(map, cmap='gray', alpha=0.5)
        ax[1, i].imshow(preds[i], cmap='jet', alpha=0.5)
        ax[1, i].set_title(f'Overlay of Original and Prediction Grid Maps {i+1}')

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

def train_ensemble(models, device, models_types):

    ensemble = EnsembleModel(len(models)).to(device)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()  # or MSELoss if smoother grayscale output

    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        loader = load_dataset('val', i, device, 8)
        print("\nLenght test dataset: ", len(loader))

        train_loss = 0

        for inputs, targets in loader:
            targets = targets.float()

            optimizer.zero_grad()
            
            outputs = [model(inputs) for model in models]  # Each output: (B, 1, H, W)
            outputs = [o.squeeze(1) for o in outputs]      # Each now: (B, H, W)
            final_output = torch.stack(outputs, dim=1)     # (B, 3, H, W)
            prediction = ensemble(final_output)            # (B, 1, H, W)
            #print("shape prediction:", prediction.shape)
            #print("shape targets:", targets.shape)
            loss = criterion(prediction, targets)          # Works fine!
            # Backpropagation
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(loader)
        print(f'Train Loss: {train_loss:.4f}')

    test_losses = []
    for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")
        test_loader = load_dataset('test', i, device, 8)
        print("\nLenght test dataset: ", len(test_loader))
        gc.collect()

        ensemble.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()

                outputs = [model(inputs) for model in models]  # Each output: (B, 1, H, W)
                outputs = [o.squeeze(1) for o in outputs]      # Each now: (B, H, W)
                final_output = torch.stack(outputs, dim=1)     # (B, 3, H, W)
                prediction = ensemble(final_output)            # (B, 1, H, W)

                # Calculate loss with true noise
                loss = criterion(prediction, targets)

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

def evaluate(models, device, ensemble_name):
    test_losses = [[] for _ in range(len(models) + 1)]
    criterion = torch.nn.BCEWithLogitsLoss()

    # Load the saved ensemble model
    model_save_path = os.path.join(MODEL_DIR, ensemble_name + '.pth')
    ensemble = EnsembleModel(len(models))  # Initialize the model
    ensemble.load_state_dict(torch.load(model_save_path, weights_only = True))  # Load the saved state dict
    ensemble.to(device)  # Move to the appropriate device

    ensemble.eval()

    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('test', i, device, 8)
        print("\nLenght test dataset: ", len(test_loader))

        test_losses_chunk = [0] * (len(models) + 1)
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.float()
                outputs = [model(inputs) for model in models]
                avg_output = sum(outputs) / len(outputs)

                outputs_squeeze = [o.squeeze(1) for o in outputs]      # Each now: (B, H, W)
                final_output = torch.stack(outputs_squeeze, dim=1)     # (B, 3, H, W)
                prediction = ensemble(final_output)            # (B, 1, H, W)

                losses = [criterion(output, targets).item() for output in outputs]
                losses.append(criterion(avg_output, targets).item())
                losses.append(criterion(prediction, targets).item())
                test_losses_chunk = [sum(x) for x in zip(test_losses_chunk, losses)]

        test_losses_chunk = [loss / len(test_loader) for loss in test_losses_chunk]
        for idx, loss in enumerate(test_losses_chunk[:-1]):
            print(f'Test Loss {idx + 1}: {loss:.4f}')
        print(f'Test Loss average: {test_losses_chunk[-2]:.4f}')
        print(f'Test Loss ensemble: {test_losses_chunk[-1]:.4f}')

        for idx, loss in enumerate(test_losses_chunk):
            test_losses[idx].append(loss)

    total_losses = [sum(losses) / len(losses) for losses in test_losses]

    print("\n----------------------------Total losses----------------------------")

    for idx, total_loss in enumerate(total_losses[:-1]):
        print(f"\nTotal loss {idx + 1}:", total_loss)
    print("\nTotal loss average:", total_losses[-2])
    print("Total loss ensemble:", total_losses[-1])

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
                predictions = []
                grid_maps = []
                gt = []
                model_predictions = [[] for _ in models]
                
                inputs, target = data
                target = target.float()
                
                outputs = [model(inputs) for model in models]
                avg_output = sum(outputs) / len(outputs)

                outputs_squeeze = [o.squeeze(1) for o in outputs]      # Each now: (B, H, W)
                final_output = torch.stack(outputs_squeeze, dim=1)     # (B, 3, H, W)
                prediction = ensemble(final_output)            # (B, 1, H, W)

                # Apply sigmoid to the outputs
                outputs = [sigmoid(output) for output in outputs]
                avg_output = sigmoid(avg_output)
                prediction = sigmoid(prediction)

                # Apply threshold to convert outputs to 0 or 1
                #threshold = 0.4
                #outputs = [(output > threshold).float() for output in outputs]
                #avg_output = (avg_output > threshold).float()
                
                predictions.append(avg_output)
                predictions.append(prediction)
                for idx, output in enumerate(outputs):
                    model_predictions[idx].append(output)
                
                predictions = torch.cat(predictions).cpu().numpy()
                model_predictions = [torch.cat(pred).cpu().numpy() for pred in model_predictions]
                grid_maps = data[0].cpu().numpy()
                gt = data[1].cpu().numpy()

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                gt = gt.reshape(-1, 400, 400)
                model_predictions = [pred.reshape(-1, 400, 400) for pred in model_predictions]

                for i in range(len(gt)):
                    visualize_prediction(predictions, gt[i], grid_maps[4], [pred[i] for pred in model_predictions])

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
        'model_20250408_180725_loss_0.0160_Autoencoder_big',
        'model_20250408_164503_loss_0.0149_Autoencoder_classic',
        'model_20250408_183632_loss_0.0164_BigUNet_autoencoder'
    ]

    ensemble_name = "Ensemble_model_20250409_180104_loss_0.0154_Autoencoder_big_Autoencoder_classic_BigUNet_autoencoder"

    models_types = [
        Autoencoder_big(),
        Autoencoder_classic(),
        BigUNet_autoencoder()
    ]

    models, device = model_preparation(model_names, models_types)

    #find_best_threshold(models, device)

    #train_ensemble(models, device, models_types)

    #evaluate(models, device, ensemble_name)

    try:

        show_predictions(models, device, ensemble_name)

    except KeyboardInterrupt:
        
        print("Program interrupted by user.")

    dist.destroy_process_group()
