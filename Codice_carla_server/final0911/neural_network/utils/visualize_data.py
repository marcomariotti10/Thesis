import os
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, RandomHorizontalFlip, RandomTranslate, Cutout
import sys
from ffcv.fields.decoders import NDArrayDecoder
import torch


def load_dataset(name, i, device):
    name_train = f"dataset_train1.beton"  # Define the path where the dataset will be written
    
    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path_train = os.path.join(complete_name_ffcv_path, name_train)

    train_loader = Loader(complete_path_train, batch_size=32,
                          num_workers=8, order=OrderOption.QUASI_RANDOM,
                          os_cache=True,
                          pipelines={
                              'covariate': [NDArrayDecoder(),    # Decodes raw NumPy arrays                    
                                            RandomHorizontalFlip(1),
                                            ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                                            ToDevice(device, non_blocking=True)],
                              'label': [NDArrayDecoder(),    # Decodes raw NumPy arrays
                                        RandomHorizontalFlip(1),
                                        ToTensor(),          # Converts to PyTorch Tensor (1,400,400)
                                        ToDevice(device, non_blocking=True)]
                          })

    return train_loader

def visualize_data(loader):
    for batch in loader:
        images, labels = batch
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        for i in range(min(20, len(images))):  # Visualize first 5 images
            
            
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(images[i].squeeze(), cmap='gray', alpha=0.5)
            ax.imshow(labels[i].squeeze(), cmap='jet', alpha=0.5)
            ax.set_title('Overlay of Original and Prediction Grid Maps')
            plt.show()


if __name__ == "__main__":

    # Get the parent directory (one level up from the current script's directory)
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Add the parent directory to sys.path
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = load_dataset('example', 1, device)
    visualize_data(loader)
