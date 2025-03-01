import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import os
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import gc
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice
from functions_for_NN import *
from constants import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp  # Use torch.multiprocessing instead

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def load_dataset(name, i, device, rank, world_size):
    name_train = f"dataset_{name}{i}.beton"  # Define the path where the dataset will be written
    complete_name_chunck_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path_train = os.path.join(complete_name_chunck_path, name_train)

    #print(f"Loading dataset from {complete_path_train}")

    dataset = Loader(complete_path_train, batch_size=32,
                     num_workers=8, order=OrderOption.SEQUENTIAL, distributed=True, seed=SEED,
                     os_cache=False,
                     pipelines={
                         'covariate': [NDArrayDecoder(),  # Decodes raw NumPy arrays
                                       ToTensor(),  # Converts to PyTorch Tensor (1,400,400)
                                       ToDevice(device, non_blocking=True)],
                         'label': [NDArrayDecoder(),  # Decodes raw NumPy arrays
                                   ToTensor(),  # Converts to PyTorch Tensor (1,400,400)
                                   ToDevice(device, non_blocking=True)]
                     })

    #print(f"Dataset {name} {i} length: {len(dataset)}")

    return dataset

def main(rank, world_size):
    setup(rank, world_size)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Model creation
    model = Autoencoder().to(rank)
    model.apply(weights_init)
    summary(model, (1,400,400))
    model = DDP(model, device_ids=[rank])
    criterion = torch.nn.BCEWithLogitsLoss().to(rank)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)

    # Check if CUDA is available
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        print(f"Using GPU: {torch.cuda.get_device_name(rank)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Parameters for training
    early_stopping_triggered = False
    number_of_chucks = NUMBER_OF_CHUNCKS
    num_total_epochs = 2
    num_epochs_for_each_chunck = 3
    number_of_chucks_testset = NUMBER_OF_CHUNCKS_TEST

    for j in range(num_total_epochs):
        if early_stopping_triggered:
            break

        print(f"\nEpoch number {j+1} of {num_total_epochs}: ")
        random.seed(SEED + j)

        for i in range(number_of_chucks):
            if early_stopping_triggered:
                break

            print(f"\nChunck number {i+1} of {number_of_chucks}")

            with ThreadPoolExecutor(max_workers=2) as executor:
                train_loader, val_loader = executor.map(load_dataset, ['train', 'val'], [i, i], [device, device], [rank, rank], [world_size, world_size])

            #train_loader = load_dataset('train', i, device, rank, world_size)
            #val_loader = load_dataset('val', i, device, rank, world_size)
            
            print("\nLength of the datasets:", len(train_loader), len(val_loader))

            for epoch in range(num_epochs_for_each_chunck):
                start = datetime.now()
                model.train()
                train_loss = 0
                for data in train_loader:
                    inputs, targets = data
                    targets = targets.float()
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data in val_loader:
                        inputs, targets = data
                        targets = targets.float()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

                scheduler.step(val_loss)
                print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                if j >= 2:
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        early_stopping_triggered = True
                        break
                else:
                    print("too early to stop")

                print(f"Time to move data to GPU: {datetime.now() - start}")

    print("\n-------------------------------------------")
    print("Training completed")
    print("-------------------------------------------\n")

    gc.collect()

    test_losses = []
    for i in range(number_of_chucks_testset):
        print(f"\nTest chunck number {i+1} of {number_of_chucks_testset}: ")

        test_loader = load_dataset('test', i, device, rank, world_size)

        print("\nLength test dataset: ", len(test_loader))

        gc.collect()

        # Evaluate on test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                targets = targets.float()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')

        test_losses.append(test_loss)

    total_loss = sum(test_losses) / len(test_losses)
    print("\nTotal loss:", total_loss)

    # Define the directory where you want to save the model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the model
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f'model_{time}_loss_{total_loss:.4f}.pth'
    model_save_path = os.path.join(MODEL_DIR, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved : {model_name}')

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print("Word size :", world_size)
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)