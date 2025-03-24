import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import os
import matplotlib.pyplot as plt
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, RandomHorizontalFlip, RandomTranslate, Cutout
import sys
from ffcv.fields.decoders import NDArrayDecoder
import torch
import torch.distributed as dist

def visualize_data(loader):
    for batch in loader:
        images, labels = batch
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()

        print(f"Images shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")

        for i in range(min(20, len(images))):  # Visualize first 5 images
            
            fig, ax = plt.subplots(2,3,figsize=(10, 10))
            ax[0,0].imshow(images[i][0], cmap='gray')
            ax[0,1].imshow(images[i][1], cmap='gray')
            ax[0,2].imshow(images[i][2], cmap='gray')
            ax[1,0].imshow(images[i][3], cmap='gray')
            ax[1,1].imshow(images[i][4], cmap='gray')
            ax[1,2].imshow(labels[i].squeeze(), cmap='jet', alpha=0.5)
            ax[1,2].imshow(images[i][4], cmap='gray', alpha=0.5)
            plt.show()


if __name__ == "__main__":

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'  # This is the address of the master node
    os.environ['MASTER_PORT'] = '29500'     # This is the port for communication (can choose any available port)

    # Set other environment variables for single-node multi-GPU setup
    os.environ['RANK'] = '0'       # Process rank (0 for single process)
    os.environ['WORLD_SIZE'] = '1'  # Total number of processes
    os.environ['LOCAL_RANK'] = '0'  # Local rank for single-GPU (0 for single GPU)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')  # Use NCCL for multi-GPU setups

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = load_dataset('train', 1, device, 16)
    visualize_data(loader)
