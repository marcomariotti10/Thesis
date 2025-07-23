import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import sys
import torch
import torch.distributed as dist
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
import os

def calculate_target_distribution(loader):
    total_zeros = 0
    total_ones = 0
    total_elements = 0

    for data in loader:
        _, targets = data
        total_zeros += (targets == 0).sum().item()
        total_ones += (targets == 1).sum().item()
        total_elements += targets.numel()

    avg_zeros = total_zeros / total_elements
    avg_ones = total_ones / total_elements

    print(f"Average number of 0s: {total_zeros} , {avg_zeros:.4f}")
    print(f"Average number of 1s: {total_ones} , {avg_ones:.4f}")

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(NUMBER_OF_CHUNCKS):
        train_loader = load_dataset('train', i, device, 16)
        calculate_target_distribution(train_loader)