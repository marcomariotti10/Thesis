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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = load_dataset('train', 1, device, 16)
    visualize_data(loader)
