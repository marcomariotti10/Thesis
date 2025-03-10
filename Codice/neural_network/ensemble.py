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
import sys
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from torch.utils.data import DataLoader, TensorDataset

def visualize_prediction(pred, gt, map, preds):
    num_preds = len(preds)
    fig, ax = plt.subplots(2, num_preds, figsize=(12, 16))
    
    ax[0, 0].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 0].imshow(pred, cmap='jet', alpha=0.5)
    ax[0, 0].set_title('Overlay of Original and Prediction Grid Maps')

    ax[0, 1].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 1].imshow(gt, cmap='jet', alpha=0.5)
    ax[0, 1].set_title('Overlay of Original and Ground Truth Grid Maps')

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

        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()
        models.append(model)
        model = model.to(device)

        # Check for multiple GPUs
        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
            model = nn.DataParallel(model)

    return models, device

def evaluate(models, device):
    test_losses = [[] for _ in range(len(models) + 1)]
    criterion = torch.nn.BCEWithLogitsLoss()

    for i in range(NUMBER_OF_CHUNCKS_TEST):
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('test', i, device, 16)
        print("\nLenght test dataset: ", len(test_loader))

        test_losses_chunk = [0] * (len(models) + 1)
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.float()
                outputs = [model(inputs) for model in models]
                avg_output = sum(outputs) / len(outputs)
                losses = [criterion(output, targets).item() for output in outputs]
                losses.append(criterion(avg_output, targets).item())
                test_losses_chunk = [sum(x) for x in zip(test_losses_chunk, losses)]

        test_losses_chunk = [loss / len(test_loader) for loss in test_losses_chunk]
        for idx, loss in enumerate(test_losses_chunk[:-1]):
            print(f'Test Loss {idx + 1}: {loss:.4f}')
        print(f'Test Loss: {test_losses_chunk[-1]:.4f}')

        for idx, loss in enumerate(test_losses_chunk):
            test_losses[idx].append(loss)

    total_losses = [sum(losses) / len(losses) for losses in test_losses]

    for idx, total_loss in enumerate(total_losses[:-1]):
        print(f"\nTotal loss {idx + 1}:", total_loss)
    print("\nTotal loss:", total_losses[-1])

def show_predictions(models, device):
    sigmoid = torch.nn.Sigmoid()

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

                # Apply sigmoid to the outputs
                outputs = [sigmoid(output) for output in outputs]
                avg_output = sigmoid(avg_output)
                
                predictions.append(avg_output)
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

                for i in range(len(grid_maps)):
                    visualize_prediction(predictions[i], gt[i], grid_maps[i], [pred[i] for pred in model_predictions])

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
        'model_20250303_015149_loss_0.0016_big_encoder',
        'model_20250301_181025_loss_0.0020_BCEwithlogitloss',
        'model_20250304_025728_loss_0.0012_unet'
    ]

    models_types = [
        Autoencoder_big(),
        Autoencoder_classic(),
        UNet()
    ]

    models, device = model_preparation(model_names, models_types)

    evaluate(models, device)

    show_predictions(models, device)

    dist.destroy_process_group()
