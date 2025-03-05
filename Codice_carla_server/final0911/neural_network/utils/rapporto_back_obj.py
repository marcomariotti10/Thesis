import sys
import torch
import torch.distributed as dist
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
import os

def load_dataset(name, i, device):
    
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written    
    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path_train = os.path.join(complete_name_ffcv_path, name_train)

    train_loader = Loader(complete_path_train, batch_size=32,
                          num_workers=8, order=OrderOption.QUASI_RANDOM,
                          os_cache=True,
                          pipelines={
                              'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                                            #RandomHorizontalFlip(1),
                                            ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                                            ToDevice(device, non_blocking=True)],
                              'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                                        #RandomHorizontalFlip(1),
                                        ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                                        ToDevice(device, non_blocking=True)]
                          })

    return train_loader

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

    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(NUMBER_OF_CHUNCKS):
        train_loader = load_dataset('train', i, device)
        calculate_target_distribution(train_loader)