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

class Autoencoder_classic(nn.Module):
    def __init__(self): # Constructor method for the autoencoder
        super(Autoencoder_classic, self).__init__() # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, x): # The forward method defines the computation that happens when the model is called with input x.
        x = self.encoder(x).contiguous()
        x = self.decoder(x).contiguous()
        return x

def load_dataset(name, i, device, batch_size):
    name_train = f"dataset_{name}{i}.beton"
    complete_name_chunck_path = os.path.join(FFCV_DIR, f'{NUMBER_OF_CHUNCKS}_{NUMBER_OF_CHUNCKS_TEST}')
    complete_path_train = os.path.join(complete_name_chunck_path, name_train)
    
    train_loader = Loader(complete_path_train, batch_size=batch_size, num_workers=8, order=OrderOption.RANDOM,
                          distributed=True, seed=SEED, drop_last=True, os_cache=False,
                          pipelines={
                              'covariate': [NDArrayDecoder(), ToTensor(), ToDevice(device, non_blocking=True)],
                              'label': [NDArrayDecoder(), ToTensor(), ToDevice(device, non_blocking=True)]
                          })
    return train_loader

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    dist.init_process_group(backend='nccl')

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(parent_dir)

    from functions_for_NN import *
    from constants import *

    gc.collect()
    random.seed(SEED)
    
    batch_sizes = [4, 8, 16, 32, 64, 128]
    learning_rates = [0.1, 0.001, 0.005]
    test_results = {}
    
    for batch_size in batch_sizes:
        for lr in learning_rates:
            print(f"Running training with batch_size={batch_size} and learning_rate={lr}")
            model = Autoencoder_classic()
            model.apply(weights_init)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            num_epochs = 50
            num_chunks = NUMBER_OF_CHUNCKS
            early_stopping_triggered = False
            
            for j in range(num_epochs):
                if early_stopping_triggered:
                    break
                
                for i in range(num_chunks):
                    if early_stopping_triggered:
                        break
                    
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        train_loader, val_loader = executor.map(load_dataset, ['train', 'val'], [i, i], [device, device], [batch_size, batch_size])
                    
                    for epoch in range(3):
                        model.train()
                        train_loss = 0
                        for data in train_loader:
                            inputs, targets = data
                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, targets.float())
                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                        train_loss /= len(train_loader)
                        
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for data in val_loader:
                                inputs, targets = data
                                outputs = model(inputs)
                                loss = criterion(outputs, targets.float())
                                val_loss += loss.item()
                        val_loss /= len(val_loader)
                        
                        scheduler.step(val_loss)
                        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
                        
                        if j >= 2:
                            early_stopping(val_loss)
                            if early_stopping.early_stop:
                                early_stopping_triggered = True
                                break
            
            test_losses = []
            for i in range(NUMBER_OF_CHUNCKS_TEST):
                test_loader = load_dataset('test', i, device, batch_size)
                model.eval()
                test_loss = 0
                with torch.no_grad():
                    for data in test_loader:
                        inputs, targets = data
                        outputs = model(inputs)
                        loss = criterion(outputs, targets.float())
                        test_loss += loss.item()
                test_loss /= len(test_loader)
                test_losses.append(test_loss)
                
            avg_test_loss = sum(test_losses) / len(test_losses)
            test_results[(batch_size, lr)] = avg_test_loss
            print(f"Test loss for batch_size={batch_size}, learning_rate={lr}: {avg_test_loss:.4f}")
    
    print("\nFinal Results:")
    for key, value in test_results.items():
        print(f"Batch Size {key[0]}, Learning Rate {key[1]} -> Test Loss: {value:.4f}")
