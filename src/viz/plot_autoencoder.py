"""
Multi-Head Autoencoder Results Evaluation and Visualization

This module provides comprehensive evaluation tools for multi-head autoencoder models trained on LiDAR data.
It implements metrics calculation, center-based matching algorithms, and visualization capabilities for 
analyzing model performance across multiple prediction heads.

Key Features:
- Pixel-level classification metrics (IoU, F1, Recall, Precision)
- Center-based object matching with distance thresholds
- Multi-head evaluation with per-head and averaged metrics
- Interactive prediction visualization with overlay capabilities
- Distributed training support for multi-GPU setups

The evaluation process handles variable numbers of prediction heads dynamically and provides both
quantitative metrics and qualitative visualizations for thorough model assessment.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import argparse
import torch.distributed as dist
from torch import nn
from typing import Optional
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
from torchsummary import summary

def _fmt4(x: float) -> str:
    """Format floats to 4 decimals; show '-' for NaN."""
    return "-" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

def _extract_box_centers(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract object centers from a binary mask using connected component analysis.
    
    This function identifies connected components in a binary mask and computes the center
    of each component's bounding box. It's used for object detection evaluation where
    we need to match predicted objects with ground truth objects based on their centers.
    
    Args:
        binary_mask (np.ndarray): 2D binary array where 1 indicates object presence and 0 background.
                                 Expected shape: (height, width)
    
    Returns:
        np.ndarray: Array of object centers as (row, col) coordinates.
                   Shape: (num_objects, 2) where each row is [row_center, col_center]
                   Returns empty array if no objects found.
    
    Process:
        1. Convert input to uint8 if needed for scipy operations
        2. Apply connected component labeling to identify separate objects
        3. For each component, find its bounding box and compute center
        4. Return centers as floating-point coordinates for sub-pixel precision
    """
    # Ensure proper data type for scipy operations
    if binary_mask.dtype != np.uint8:
        mask = (binary_mask > 0).astype(np.uint8)
    else:
        mask = binary_mask
    
    # Find connected components (separate objects)
    lbl, n = ndi.label(mask)
    if n == 0:
        return np.empty((0, 2), dtype=float)

    # Extract bounding box for each connected component
    objects = ndi.find_objects(lbl)
    centers = []
    for slc in objects:
        if slc is None:
            continue
        # Calculate bounding box center coordinates
        r0, r1 = slc[0].start, slc[0].stop
        c0, c1 = slc[1].start, slc[1].stop
        centers.append([(r0 + r1 - 1) / 2.0, (c0 + c1 - 1) / 2.0])  # (row, col)
    
    return np.asarray(centers, dtype=float)


def _match_centers(gt_centers: np.ndarray,
                   pred_centers: np.ndarray,
                   meters_per_pixel: float = 1.0,
                   max_match_distance_m: Optional[float] = None):
    """
    Match predicted object centers with ground truth centers using optimal assignment.
    
    This function implements the Hungarian algorithm to find the optimal pairing between
    predicted and ground truth object centers, minimizing the total matching distance.
    Used for evaluating object detection performance in spatial domains.
    
    Args:
        gt_centers (np.ndarray): Ground truth object centers, shape (num_gt, 2)
                                Each row is [row, col] in pixel coordinates
        pred_centers (np.ndarray): Predicted object centers, shape (num_pred, 2)
                                  Each row is [row, col] in pixel coordinates
        meters_per_pixel (float): Conversion factor from pixels to meters
                                 Default: 1.0 (assumes pixel = meter)
        max_match_distance_m (Optional[float]): Maximum allowed matching distance in meters
                                              Pairs beyond this distance are rejected
                                              Default: None (no distance limit)
    
    Returns:
        tuple: (distances_m, num_gt_unmatched, num_pred_unmatched)
            - distances_m (np.ndarray): 1D array of matched pair distances in meters
            - num_gt_unmatched (int): Number of ground truth objects without matches
            - num_pred_unmatched (int): Number of predicted objects without matches
    
    Algorithm:
        1. Handle edge cases (empty inputs)
        2. Compute pairwise distances in meters
        3. Apply distance threshold if specified
        4. Use Hungarian algorithm for optimal assignment
        5. Filter out pairs exceeding distance threshold
        6. Count unmatched objects in both sets
    """
    G, P = len(gt_centers), len(pred_centers)
    
    # Handle edge cases with empty inputs
    if G == 0 and P == 0:
        return np.array([], dtype=float), 0, 0
    if G == 0:
        return np.array([], dtype=float), 0, P
    if P == 0:
        return np.array([], dtype=float), G, 0

    # Compute pairwise distances in meters
    gt = gt_centers[:, None, :]     # (G,1,2) - broadcast for pairwise computation
    pr = pred_centers[None, :, :]   # (1,P,2) - broadcast for pairwise computation
    d_pix = np.linalg.norm(gt - pr, axis=-1)  # (G,P) - Euclidean distances in pixels
    d_m = d_pix * meters_per_pixel

    # Prepare cost matrix for Hungarian algorithm
    cost = d_m.copy()
    BIG = 1e9  # Large value to mark invalid assignments
    if max_match_distance_m is not None:
        cost[cost > max_match_distance_m] = BIG

    # Apply Hungarian algorithm for optimal assignment
    rows, cols = linear_sum_assignment(cost)
    
    # Filter out assignments that exceed distance threshold
    valid = cost[rows, cols] < BIG
    rows, cols = rows[valid], cols[valid]
    d_keep = d_m[rows, cols]

    # Count unmatched objects
    num_gt_unmatched = G - len(rows)
    num_pred_unmatched = P - len(cols)
    
    return d_keep, num_gt_unmatched, num_pred_unmatched

def visualize_prediction(pred, gt, map_img, head_idx):
    """
    Visualize model predictions alongside ground truth with background map overlay.
    
    This function creates a side-by-side comparison of predicted and ground truth object
    positions overlaid on the original LiDAR grid map. Useful for qualitative assessment
    of model performance and understanding prediction patterns.
    
    Args:
        pred (np.ndarray): Predicted binary mask, shape (height, width)
                          Values should be 0 (background) or 1 (object)
        gt (np.ndarray): Ground truth binary mask, shape (height, width)
                        Values should be 0 (background) or 1 (object)
        map_img (np.ndarray): Background LiDAR grid map, shape (height, width)
                             Used as grayscale base layer for context
        head_idx (int): Index of the prediction head (0-based)
                       Used for labeling in visualization title

    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    
    # Left panel: Prediction overlay
    ax[0].imshow(map_img, cmap='gray', alpha=0.5)  # Background map
    ax[0].imshow(pred, cmap='jet', alpha=0.5)      # Prediction overlay
    ax[0].set_title(f'Head {head_idx+1} - Prediction Overlay')

    # Right panel: Ground truth overlay
    ax[1].imshow(map_img, cmap='gray', alpha=0.5)  # Background map
    ax[1].imshow(gt, cmap='jet', alpha=0.5)        # Ground truth overlay
    ax[1].set_title(f'Head {head_idx+1} - Ground Truth Overlay')
    
    plt.show()

def compute_metrics_from_confusion_matrix(cm):
    """
    Calculate comprehensive classification metrics from a confusion matrix.
    
    This function computes standard binary classification metrics used for evaluating
    object detection and segmentation performance. All metrics handle edge cases
    to avoid division by zero errors.
    
    Args:
        cm (np.ndarray): 2x2 confusion matrix with structure:
                        [[TN, FP],
                         [FN, TP]]
                        where TN=True Negative, FP=False Positive,
                        FN=False Negative, TP=True Positive
    
    Returns:
        dict: Dictionary containing calculated metrics:
            - 'Accuracy': (TP + TN) / (TP + TN + FP + FN)
                         Overall classification accuracy
            - 'Precision': TP / (TP + FP)
                          Proportion of positive predictions that are correct
            - 'Recall': TP / (TP + FN)
                       Proportion of actual positives that are detected
            - 'F1 Score': 2 * (Precision * Recall) / (Precision + Recall)
                         Harmonic mean of precision and recall
            - 'IoU': TP / (TP + FP + FN)
                    Intersection over Union for object overlap
    
    Note:
        All metrics return 0.0 if their denominator is zero (no relevant samples).
        This prevents division by zero errors in edge cases.
    """
    # Extract confusion matrix components
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # Calculate metrics with zero-division protection
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # Total always > 0 in practice
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'IoU': iou
    }

def model_preparation(model_name, model_type):
    """
    Prepare and load a trained neural network model for evaluation.
    
    This function handles the complete model setup process including CUDA detection,
    model instantiation, checkpoint loading, state dictionary cleaning, and multi-GPU
    configuration. It ensures the model is properly loaded and ready for inference.
    
    Args:
        model_name (str): Name of the saved model file (without .pth extension)
                         Used to locate the model checkpoint in MODEL_DIR
        model_type (class): Model class constructor (e.g., LargeModel, MediumModel)
                           Used to instantiate the model architecture
    
    Returns:
        tuple: (model, device)
            - model: Loaded PyTorch model ready for inference
                    Wrapped in DataParallel if multiple GPUs available
            - device: PyTorch device object (cuda or cpu)
                     Used for tensor operations and model placement
    
    Process:
        1. Detect and display CUDA capabilities
        2. Instantiate model from provided class
        3. Load checkpoint and clean state dictionary
        4. Configure device placement (GPU/CPU)
        5. Setup multi-GPU training if available
        6. Display model summary for verification
    """
    # Display CUDA and hardware information
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    # Get current CUDA device information
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    # Instantiate model architecture
    model = model_type()
    
    # Construct full path to model checkpoint
    model_path = MODEL_DIR
    model_name = model_name + '.pth'
    model_path = os.path.join(model_path, model_name)
    
    # Configure device (GPU if available, otherwise CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Extract state dictionary (handle different checkpoint formats)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Clean state dictionary by removing 'module.' prefix if present
    # This handles compatibility between single-GPU and multi-GPU trained models
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    # Load cleaned state dictionary into model
    model.load_state_dict(new_state_dict)

    # Move model to appropriate device
    model = model.to(device)

    # Configure multi-GPU setup if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
        model = nn.DataParallel(model)

    # Display model architecture summary
    if isinstance(model, nn.DataParallel):
        summary(model.module, input_size=(NUMBER_FRAMES_INPUT, 400, 400))
    else:
        summary(model, input_size=(NUMBER_FRAMES_INPUT, 400, 400))

    return model, device

def evaluate_multi_head(model, device):
    """
    Comprehensive evaluation of multi-head autoencoder model performance.

    This function evaluates a multi-head model across all test data chunks, computing both
    pixel-level classification metrics and center-based object matching metrics. It handles
    variable numbers of prediction heads dynamically and provides detailed per-head and
    averaged performance statistics.

    Args:
        model: PyTorch model with multiple prediction heads
               Expected to return list/tuple of head outputs or tensor with head dimension
        device: PyTorch device for tensor operations (cuda or cpu)

    Output Format:
        Prints metrics for each head in order: IoU, F1, Recall, Precision, Mean Distance, Median Distance
        Finally prints averaged metrics across all heads in the same order

    Metrics Computed:
        - Classification Metrics (pixel-level):
            * IoU: Intersection over Union for object overlap
            * F1 Score: Harmonic mean of precision and recall  
            * Recall: Proportion of actual positives detected
            * Precision: Proportion of positive predictions that are correct
        - Distance Metrics (object-level):
            * Mean Distance: Average distance between matched object pairs (meters)
            * Median Distance: Median distance between matched object pairs (meters)

    Configuration:
        - meters_per_pixel: Conversion factor from pixels to real-world meters
        - max_match_distance_m: Maximum distance for valid object matching
        - Uses Hungarian algorithm for optimal object pairing

    Process:
        1. Initialize per-head accumulators for confusion matrices and distances
        2. Iterate through all test data chunks
        3. For each batch: run inference, extract predictions, accumulate metrics
        4. For each head: compute classification metrics and distance statistics
        5. Print per-head results and compute/print averaged results
    """
    # Configuration for center matching (can be overridden by global constants)
    meters_per_pixel = globals().get("METERS_PER_PIXEL", GRID_RESOLUTION)
    max_match_distance_m: Optional[float] = globals().get("MAX_MATCH_DISTANCE_M", 2.0)

    # Per-head accumulators (initialized lazily after seeing first batch)
    head_conf_matrices = None                  # list of (2,2) confusion matrices
    head_all_distances = None                  # list of lists of matched distances

    model.eval()

    with torch.no_grad():
        # Process all test data chunks
        for chunk_idx in range(NUMBER_OF_CHUNCKS_TEST):
            test_loader = load_dataset('test', chunk_idx, device, BATCH_SIZE)

            for inputs, targets in test_loader:
                num_heads = targets.shape[1]

                # Lazy initialization of per-head accumulators
                if head_conf_matrices is None:
                    head_conf_matrices = [np.zeros((2, 2), dtype=int) for _ in range(num_heads)]
                    head_all_distances = [[] for _ in range(num_heads)]

                # Split targets into per-head tensors: [B,1,H,W] each
                t_list = [targets[:, h].unsqueeze(1).float().to(device) for h in range(num_heads)]

                # Forward pass - handle different output formats
                outputs = model(inputs)

                # Convert outputs to list format for uniform processing
                if isinstance(outputs, (list, tuple)):
                    o_list = list(outputs)
                elif torch.is_tensor(outputs) and outputs.ndim >= 5 and outputs.shape[1] == num_heads:
                    # Split tensor outputs into list: [B,1,H,W] each
                    o_list = [outputs[:, h] for h in range(num_heads)]
                else:
                    raise ValueError(
                        "Model outputs must be a list/tuple of per-head logits, "
                        f"or a tensor with head dimension matching targets. Got type={type(outputs)}"
                    )

                # Process each prediction head
                for i in range(num_heads):
                    logits = o_list[i]
                    probs = torch.sigmoid(logits)          # Convert logits to probabilities
                    preds = (probs > 0.5).int()            # Binarize predictions [B,1,H,W]

                    # Convert to numpy for metric computation
                    preds_np = preds.detach().cpu().numpy().astype(int).reshape(-1, 400, 400)
                    targets_np = t_list[i].detach().cpu().numpy().astype(int).reshape(-1, 400, 400)

                    # Accumulate pixel-level confusion matrix
                    head_conf_matrices[i] += confusion_matrix(
                        targets_np.reshape(-1),      # Flatten to 1D
                        preds_np.reshape(-1),        # Flatten to 1D
                        labels=[0, 1]                # Ensure binary classification
                    )

                    # Accumulate object-level distance metrics
                    for b in range(preds_np.shape[0]):  # Process each sample in batch
                        gt_mask = targets_np[b]
                        pr_mask = preds_np[b]

                        # Extract object centers from binary masks
                        gt_centers = _extract_box_centers(gt_mask)
                        pr_centers = _extract_box_centers(pr_mask)

                        # Match centers and compute distances
                        d_keep, _, _ = _match_centers(
                            gt_centers, pr_centers,
                            meters_per_pixel=meters_per_pixel,
                            max_match_distance_m=max_match_distance_m
                        )
                        
                        # Store matched distances for statistics
                        if len(d_keep):
                            head_all_distances[i].extend(d_keep.tolist())

    # Handle case where no data was processed
    if head_conf_matrices is None or head_all_distances is None:
        return

    per_head_values = []  # list of tuples: (IoU, F1, recall, precision, mean_d, median_d)

    for i in range(len(head_conf_matrices)):
        m = compute_metrics_from_confusion_matrix(head_conf_matrices[i])

        dists = np.asarray(head_all_distances[i], dtype=float)
        mean_d = float(np.mean(dists)) if dists.size > 0 else float('nan')
        med_d  = float(np.median(dists)) if dists.size > 0 else float('nan')

        values = (
            float(m['IoU']),
            float(m['F1 Score']),
            float(m['Recall']),
            float(m['Precision']),
            mean_d,
            med_d
        )
        per_head_values.append(values)

    # Compute column-wise averages with NaN handling (avoid RuntimeWarning on all-NaN)
    arr = np.array(per_head_values, dtype=float)  # shape [num_heads, 6]
    avg = []
    for col in range(arr.shape[1]):
        col_vals = arr[:, col]
        if np.isfinite(col_vals).any():
            avg.append(float(np.nanmean(col_vals)))
        else:
            avg.append(float('nan'))

    headers = [
        "Result",
        "IoU",
        "F1",
        "Recall",
        "Precision",
        "Mean Dist (m)",
        "Median Dist (m)"
    ]

    # Build rows: one per head + final average row
    table_rows = []
    for idx, vals in enumerate(per_head_values, start=1):
        row = [
            f"Head {idx}",
            _fmt4(vals[0]),
            _fmt4(vals[1]),
            _fmt4(vals[2]),
            _fmt4(vals[3]),
            _fmt4(vals[4]),
            _fmt4(vals[5]),
        ]
        table_rows.append(row)

    table_rows.append([
        "Average",
        _fmt4(avg[0]),
        _fmt4(avg[1]),
        _fmt4(avg[2]),
        _fmt4(avg[3]),
        _fmt4(avg[4]),
        _fmt4(avg[5]),
    ])

    # Determine column widths
    col_widths = [len(h) for h in headers]
    for row in table_rows:
        for c, cell in enumerate(row):
            col_widths[c] = max(col_widths[c], len(str(cell)))

    # Pretty print
    def _pline(parts):
        print(" | ".join(str(p).ljust(col_widths[i]) for i, p in enumerate(parts)))

    def _sep():
        print("-+-".join("-" * w for w in col_widths))

    print()  # blank line before table
    _pline(headers)
    _sep()
    for r in table_rows:
        _pline(r)
    print()  # blank line after table


def show_predictions_multihead(model, device):
    """
    Interactive visualization of multi-head model predictions for qualitative analysis.
    
    This function displays model predictions alongside ground truth for visual inspection
    across all test data chunks. Each prediction head is visualized separately, allowing
    for detailed analysis of temporal prediction patterns and model behavior.
    
    Args:
        model: PyTorch multi-head model for inference
               Expected to return list of outputs, one per prediction head
        device: PyTorch device for tensor operations (cuda or cpu)
    
    Visualization Features:
        - Side-by-side comparison of predictions vs ground truth
        - Background LiDAR map overlay for spatial context
        - Interactive matplotlib windows for detailed inspection
        - Sequential processing through all test chunks
        - Per-head temporal analysis (multiple future time steps)
    
    Process:
        1. Process each test data chunk sequentially
        2. Load test data with batch size 1 for detailed visualization
        3. Run model inference and apply sigmoid activation
        4. Split targets by prediction head (temporal steps)
        5. Visualize each head's predictions interactively
        6. Allow user interaction through matplotlib interface
    
    Display Format:
        - Each head represents a different future time step
        - Overlays predictions and ground truth on original LiDAR maps
        - Color coding: jet colormap for objects, grayscale for background
        - Interactive zoom and pan capabilities for detailed inspection
    
    Data Flow:
        Input LiDAR sequences → Multi-head model → Per-head predictions → Visualization
        Each head predicts object positions at specific future time intervals
    """
    sigmoid = torch.nn.Sigmoid()
    model.eval()

    # Process all test data chunks
    for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
        # Use batch size 1 for detailed individual sample visualization
        test_loader = load_dataset('test', i, device, 1)

        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data

                # Split targets by prediction head (future time steps)
                # Each head predicts objects at different temporal offsets
                targets = [
                    targets[:, h].unsqueeze(1).float().to(device)
                    for h in range(len(FUTURE_TARGET_RILEVATION))
                ]

                # Forward pass: model returns list of head outputs
                outputs = model(inputs)
                # Convert logits to probabilities for visualization
                outputs = [sigmoid(o) for o in outputs]

                # Convert input LiDAR sequence to numpy for visualization
                inputs_np = inputs.cpu().numpy().reshape(-1, 400, 400)

                # Visualize each prediction head
                for head_idx, (out, tgt) in enumerate(zip(outputs, targets)):
                    # Convert predictions and targets to numpy
                    predictions_np = out.cpu().numpy().reshape(-1, 400, 400)
                    targets_np = tgt.cpu().numpy().reshape(-1, 400, 400)

                    # Visualize each sample in the batch (batch size = 1 here)
                    for j in range(len(predictions_np)):
                        visualize_prediction(predictions_np[j], targets_np[j], inputs_np[j], head_idx)

if __name__ == '__main__':
    
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description='Evaluate Multi-Head Model')
    parser.add_argument('--plot_type', type=str, choices=['inference', 'metrics'], default='metrics',
                       help='Type of plot to generate: inference or metrics')
    parser.add_argument('--model_type', type=str, default='MediumModel', 
                       help='Model class name (SmallModel, MediumModel, LargeModel)')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name/filename')
    args = parser.parse_args()
    
    model_mapping = {
        'SmallModel': SmallModel,
        'MediumModel': MediumModel,
        'LargeModel': LargeModel
    }
    
    if args.model_type not in model_mapping:
        print(f"Error: Unknown model type '{args.model_type}'. Available types: {list(model_mapping.keys())}")
        sys.exit(1)
    
    model_type = model_mapping[args.model_type]
    model_name = args.model_name
    plot_type = args.plot_type
    
    print(f"Running visualization for:")
    print(f"  - Model: {model_name}")
    print(f"  - Type: {args.model_type}")
    print(f"  - Plot: {plot_type}")

    # Configure distributed training environment for single-node multi-GPU setup
    os.environ['MASTER_ADDR'] = 'localhost'  # Master node address
    os.environ['MASTER_PORT'] = '29500'      # Communication port (adjust if needed)
    os.environ['RANK'] = '0'                 # Process rank (single process)
    os.environ['WORLD_SIZE'] = '1'           # Total number of processes
    os.environ['LOCAL_RANK'] = '0'           # Local rank for single-GPU

    # Initialize distributed process group for multi-GPU coordination
    dist.init_process_group(backend='nccl')  # Use NCCL for GPU communication

    # Set random seed for reproducible results
    random.seed(SEED)

    # Load and prepare model for evaluation
    model, device = model_preparation(model_name, model_type)

    try:
        if plot_type == "metrics":
            # Run comprehensive evaluation with metrics computation
            evaluate_multi_head(model, device)
        elif plot_type == "inference":
            # Show interactive prediction visualizations
            show_predictions_multihead(model, device)

    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
    finally:
        # Cleanup distributed resources
        torch.distributed.destroy_process_group()
