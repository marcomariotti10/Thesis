import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from neural_network import *
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import sys
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(name, i, device, batch):
    name_train = f"dataset_{name}{i}.beton"
    complete_name_ffcv_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path_train = os.path.join(complete_name_ffcv_path, name_train)

    train_loader = Loader(complete_path_train, batch_size=batch,
                          num_workers=8, order=OrderOption.QUASI_RANDOM,
                          os_cache=True,
                          pipelines={
                              'covariate': [NDArrayDecoder(),
                                            ToTensor(),
                                            ToDevice(device, non_blocking=True)],
                              'label': [NDArrayDecoder(),
                                        ToTensor(),
                                        ToDevice(device, non_blocking=True)]
                          })

    return train_loader

def visualize_prediction(pred, gt, map, pred1, pred2, pred3):
    fig, ax = plt.subplots(2, 3, figsize=(12, 16))
    
    ax[0, 0].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 0].imshow(pred, cmap='jet', alpha=0.5)
    ax[0, 0].set_title('Overlay of Original and Prediction Grid Maps')

    ax[0, 2].imshow(map, cmap='gray', alpha=0.5)
    ax[0, 2].imshow(gt, cmap='jet', alpha=0.5)
    ax[0, 2].set_title('Overlay of Original and Ground Truth Grid Maps')

    ax[1, 0].imshow(map, cmap='gray', alpha=0.5)
    ax[1, 0].imshow(pred1, cmap='jet', alpha=0.5)
    ax[1, 0].set_title('Overlay of Original and Prediction Grid Maps 1')

    ax[1, 1].imshow(map, cmap='gray', alpha=0.5)
    ax[1, 1].imshow(pred2, cmap='jet', alpha=0.5)
    ax[1, 1].set_title('Overlay of Original and Prediction Grid Maps 2')

    ax[1, 2].imshow(map, cmap='gray', alpha=0.5)
    ax[1, 2].imshow(pred3, cmap='jet', alpha=0.5)
    ax[1, 2].set_title('Overlay of Original and Prediction Grid Maps 3')

    plt.show()

def model_preparation(model_name_1, model_name_2, model_name_3):
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")
    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

    model_path_1 = os.path.join(MODEL_DIR, model_name_1 + '.pth')
    model_path_2 = os.path.join(MODEL_DIR, model_name_2 + '.pth')
    model_path_3 = os.path.join(MODEL_DIR, model_name_3 + '.pth')

    model_1 = Autoencoder_big()
    model_2 = Autoencoder_classic()
    model_3 = UNet()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_1 = model_1.to(device)
        model_2 = model_2.to(device)
        model_3 = model_3.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    checkpoint_1 = torch.load(model_path_1, map_location=device)
    checkpoint_2 = torch.load(model_path_2, map_location=device)
    checkpoint_3 = torch.load(model_path_3, map_location=device)

    state_dict_1 = checkpoint_1['state_dict'] if 'state_dict' in checkpoint_1 else checkpoint_1
    state_dict_2 = checkpoint_2['state_dict'] if 'state_dict' in checkpoint_2 else checkpoint_2
    state_dict_3 = checkpoint_3['state_dict'] if 'state_dict' in checkpoint_3 else checkpoint_3

    new_state_dict_1 = {key.replace('module.', ''): value for key, value in state_dict_1.items()}
    new_state_dict_2 = {key.replace('module.', ''): value for key, value in state_dict_2.items()}
    new_state_dict_3 = {key.replace('module.', ''): value for key, value in state_dict_3.items()}

    model_1.load_state_dict(new_state_dict_1)
    model_2.load_state_dict(new_state_dict_2)
    model_3.load_state_dict(new_state_dict_3)

    model_1.eval()
    model_2.eval()
    model_3.eval()

    return model_1, model_2, model_3, device

def evaluate(model_1, model_2, model_3, device):
    test_losses = []
    criterion = torch.nn.BCEWithLogitsLoss()

    for i in range(NUMBER_OF_CHUNCKS):
        print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")

        test_loader = load_dataset('train', i, device, 32)
        print("\nLenght test dataset: ", len(test_loader))

        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()
                outputs_1 = model_1(inputs)
                outputs_2 = model_2(inputs)
                outputs_3 = model_3(inputs)
                outputs = (outputs_1 + outputs_2 + outputs_3) / 3
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)

def show_predictions(model_1, model_2, model_3, device):
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
                predictions1 =[]
                predictions2 = []
                predictions3 = []
                
                inputs, target = data
                target = target.float()
                
                outputs_1 = model_1(inputs)
                outputs_2 = model_2(inputs)
                outputs_3 = model_3(inputs)
                outputs = (outputs_1 + outputs_2 + outputs_3) / 3

                # Apply sigmoid to the outputs
                outputs = sigmoid(outputs)
                outputs_1 = sigmoid(outputs_1)
                outputs_2 = sigmoid(outputs_2)
                outputs_3 = sigmoid(outputs_3)
                
                predictions.append(outputs)
                predictions1.append(outputs_1)
                predictions2.append(outputs_2)
                predictions3.append(outputs_3)
                
                predictions = torch.cat(predictions).cpu().numpy()
                predictions1 = torch.cat(predictions1).cpu().numpy()
                predictions2 = torch.cat(predictions2).cpu().numpy()
                predictions3 = torch.cat(predictions3).cpu().numpy()
                grid_maps = data[0].cpu().numpy()
                gt = data[1].cpu().numpy()

                grid_maps = grid_maps.reshape(-1, 400, 400)
                predictions = predictions.reshape(-1, 400, 400)
                gt = gt.reshape(-1, 400, 400)
                predictions1 = predictions1.reshape(-1, 400, 400)
                predictions2 = predictions2.reshape(-1, 400, 400)
                predictions3 = predictions3.reshape(-1, 400, 400)

                for i in range(len(grid_maps)):
                    visualize_prediction(predictions[i], gt[i], grid_maps[i], predictions1[i], predictions2[i], predictions3[i])

if __name__ == '__main__':

    model_name_1 = 'model_20250303_015149_loss_0.0016_big_encoder'
    model_name_2 = 'model_20250301_181025_loss_0.0020_BCEwithlogitloss'
    model_name_3 = 'model_20250304_025728_loss_0.0012_unet'

    model1, model2, model3, device = model_preparation(model_name_1, model_name_2, model_name_3)

    #evaluate(model1, model2, model3, device)

    show_predictions(model1, model2, model3, device)
