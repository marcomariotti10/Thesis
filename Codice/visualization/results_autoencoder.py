import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import argparse
import numpy as np
import os
import torch.distributed as dist
from torch import nn
import pickle
from matplotlib import gridspec
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, jaccard_score
)
import seaborn as sns
import matplotlib.pyplot as plt

def model_preparation(model_name, model_type, activation_function):

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    model = model_type(activation_fn = activation_function)
    # Load model
    model_path = MODEL_DIR
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
        
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Remove 'module.' prefix from the state dict keys if it's there
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Create a new state dict without the "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    #print("Saved model keys:", new_state_dict.keys())
    #print("Current model keys:", model.state_dict().keys())

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
    
    model.eval()

    return model, device

def compute_metrics_from_confusion_matrix(cm):
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    f1_score  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'IoU': iou
    }

def plot_conf_matrix_and_metrics(cmat, metrics, normalize=True):
    # Swap the label order: Occupied (1), Free (0)
    labels = ['Actor', 'Background']

    # Reorder the confusion matrix: put row 1 before row 0, and column 1 before column 0
    cm = cmat.astype(float)
    cm = cm[[1, 0], :]   # rows
    cm = cm[:, [1, 0]]   # columns

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm / np.where(row_sums == 0, 1, row_sums)
        fmt = ".3f"
    else:
        fmt = ".0f"

    # Create grid with width ratios: 3 for matrix, 1 for metrics
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

    # ---- Confusion Matrix ----
    ax0 = plt.subplot(gs[0])
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax0)
    ax0.set_xlabel("PREDICTED LABEL")
    ax0.set_ylabel("TRUE LABEL")
    ax0.set_title("Normalized Confusion Matrix")

    # ---- Metric Bar Chart ----
    ax1 = plt.subplot(gs[1])
    metric_names = list(metrics.keys())
    metric_values = np.array([metrics[k] for k in metric_names])

    # Color bars based on value
    norm = plt.Normalize(0, 1)
    colormap = get_cmap('Blues')
    colors = colormap(norm(metric_values))

    bars = ax1.barh(metric_names, metric_values, color=colors, edgecolor='black')
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Score")
    ax1.set_title("Metrics")

    # Add value *inside* each bar
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        bar_width = bar.get_width()
        text_color = 'white' if bar_width > 0.3 else 'black'  # optionally use 'white' for dark bars
        ax1.text(bar_width / 2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", ha='center', va='center', fontsize=9, color=text_color)

    plt.tight_layout()
    plt.show()

def evaluate(model, device):
    test_losses = []
    cmat_total = np.zeros((2, 2), dtype=int)

    criterion = torch.nn.BCEWithLogitsLoss()

    for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        print(f"\nTest chunk number {i + 1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('test', i, device, BATCH_SIZE)
        print("Length test dataset:", len(test_loader))

        gc.collect()
        model.eval()

        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.float()
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                test_losses.append(loss.item())

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                preds_np = preds.cpu().numpy().astype(int).flatten()
                targets_np = targets.cpu().numpy().astype(int).flatten()

                # Accumulate confusion matrix
                cmat_total += confusion_matrix(targets_np, preds_np, labels=[0, 1])

    # Calculate final metrics
    metrics = compute_metrics_from_confusion_matrix(cmat_total)
    avg_loss = np.mean(test_losses)

    print("\nEvaluation Results:")
    print(f"Average Test Loss  : {avg_loss:.4f}")
    for k, v in metrics.items():
        print(f"{k:17}: {v:.4f}")

    print("\nConfusion Matrix (Summed Over All Chunks):")
    print(cmat_total)

    # Plot confusion matrix + metrics
    plot_conf_matrix_and_metrics(cmat_total, metrics, normalize=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and test a neural network model.')
    parser.add_argument('--model_type', type=str, default='Autoencoder_big', help='Type of model to use')
    parser.add_argument('--activation_function', type=str, default='ReLU', help='Activation function to apply to the model')
    parser.add_argument('--model_name', type=str, default='model_20250716_211844_loss_0.0055_Autoencoder_big', help='Name of the model to load')
    args = parser.parse_args()

    model_type = globals()[args.model_type]
    activation_function = getattr(nn, args.activation_function)  # This retrieves the activation function class
    model_name = args.model_name

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # This is the address of the master node
    os.environ['MASTER_PORT'] = '29500'     # This is the port for communication (can choose any available port)

    # Set other environment variables for single-node multi-GPU setup
    os.environ['RANK'] = '0'       # Process rank (0 for single process)
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes
    os.environ['LOCAL_RANK'] = '0'  # Local rank for single-GPU (0 for single GPU)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')  # Use NCCL for multi-GPU setups
    
    model, device = model_preparation(model_name, model_type, activation_function)

    try:
        evaluate(model, device)

        #show_predictions(model, device)
    
    except KeyboardInterrupt:
        print("\n program interrupted by user")