import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import os
import gc
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch.distributed as dist
import sys
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.transforms import ToTensor, ToDevice

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    dist.init_process_group(backend='nccl')

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    gc.collect()
    random.seed(SEED)
    
    batch_sizes = [4, 8, 16, 32, 64, 128]
    learning_rates = [0.1, 0.001, 0.005]
    test_results = {}
    
    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"Running training with batch_size={batch_size} and learning_rate={lr}")
            model = Autoencoder_classic()
            model.apply(initialize_weights)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
            
            # Check if CUDA is available
            print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")

            # Check for multiple GPUs
            if torch.cuda.device_count() > 1:
                print(f"Multiple GPUs detected: {torch.cuda.device_count()}")
                model = nn.DataParallel(model)

            if torch.cuda.is_available():
                device = torch.device("cuda")
                model = model.to(device)
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("CUDA is not available. Using CPU.")

            # Storing ID of current CUDA device
            cuda_id = torch.cuda.current_device()
            print(f"ID of current CUDA device:{torch.cuda.current_device()}")
            print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
            
            # Parameters for training
            early_stopping_triggered = False
            number_of_chuncks = NUMBER_OF_CHUNCKS
            num_total_epochs = 100
            num_epochs_for_each_chunck = 1
            number_of_chuncks_val = NUMBER_OF_CHUNCKS_TEST
                    
            for j in range(num_total_epochs):
                if early_stopping_triggered:
                    break

                random.seed(SEED + j)
                train_order = random.sample(range(number_of_chuncks), number_of_chuncks)
                #print("\nOrder of the chuncks for training:", train_order)
                #print(f"\nEpoch number {j+1} of {num_total_epochs}: ")

                for i in range(number_of_chuncks):  # type: ignore
                    if early_stopping_triggered:
                        break

                    #print(f"\nChunck number {i+1} of {number_of_chuncks}")
                    train_index = train_order[i]
                    train_loader = load_dataset('train', train_index, device, batch_size)
                    #print("\nLenght of the train dataset:", len(train_loader))

                    for epoch in range(num_epochs_for_each_chunck):
                        start = datetime.now()
                        model.train()
                        train_loss = 0
                        for data in train_loader:
                            
                            inputs, targets = data
                            targets = targets.float()

                            optimizer.zero_grad()

                            # Predict the noise for this timestep
                            pred = model(inputs)

                            # Calculate loss with true noise
                            loss = criterion(pred, targets)
                            
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()

                        train_loss /= len(train_loader)

                        if epoch + 1 == num_epochs_for_each_chunck:
                            val_losses = []
                            for k in range(number_of_chuncks_val):
                                val_loader = load_dataset('val', k, device, batch_size)
                                model.eval()
                                val_loss = 0
                                with torch.no_grad():
                                    for data in val_loader:
                                        inputs, targets = data
                                        targets = targets.float()

                                        # Predict the noise for this timestep
                                        pred = model(inputs)

                                        # Calculate loss with true noise
                                        loss = criterion(pred, targets)
                    
                                        val_loss += loss.item()
                                val_loss /= len(val_loader)
                                val_losses.append(val_loss)

                            total_val_loss = sum(val_losses) / len(val_losses)
                            #print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}, Val Loss: {total_val_loss:.4f}        Time: {datetime.now() - start}')

                            if j >= 1:
                                
                                scheduler.step(val_loss)
                                early_stopping(val_loss)
                                if early_stopping.early_stop:
                                    print("Early stopping triggered")
                                    early_stopping_triggered = True
                                    break
                        else:
                            #print(f'Epoch {epoch+1}/{num_epochs_for_each_chunck}, Train Loss: {train_loss:.4f}                          Time: {datetime.now() - start}')
                            pass

            print("------------------STARTING TESTING-------------------")
            test_losses = []
            for i in range(NUMBER_OF_CHUNCKS_TEST):  # type: ignore
                print(f"\nTest chunck number {i+1} of {NUMBER_OF_CHUNCKS_TEST}: ")
                test_loader = load_dataset('test', i, device, batch_size)
                print("\nLenght test dataset: ", len(test_loader))
                gc.collect()

                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        inputs, targets = data
                        targets = targets.float()
                        
                        # Predict the noise for this timestep
                        pred = model(inputs)

                        # Calculate loss with true noise
                        loss = criterion(pred, targets)

                        test_loss += loss.item()
                test_loss /= len(test_loader)
                print(f'Test Loss: {test_loss:.4f}')
                test_losses.append(test_loss)

            total_loss = sum(test_losses) / len(test_losses)
            print(f"\nTotal loss for batch_size={batch_size}, learning_rate={lr}:", total_loss)
            test_results[(batch_size, lr)] = total_loss
            print("-------------------------------------------------")
    
    print("\nFinal Results:")
    for key, value in test_results.items():
        print(f"Batch Size {key[0]}, Learning Rate {key[1]} -> Test Loss: {value:.4f}")
    
    dist.destroy_process_group()
