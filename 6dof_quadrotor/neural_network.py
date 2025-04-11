import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


### 1. Dataset class ###

class ControlAllocationDataset(Dataset):
    def __init__(self, dataset_path, num_outputs, transform=None):
        self.dataframe = pd.read_csv(dataset_path, header = None).astype('float32')
        self.transform = transform
        self.num_outputs = num_outputs
        self.num_inputs = np.shape(self.dataframe)[1] - num_outputs
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input = self.dataframe.iloc[idx, 0:self.num_inputs]
        output = self.dataframe.iloc[idx, self.num_inputs:]

        sample = {'input': np.array(input), 'output': np.array(output)}

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample

### 2. Neural Network class ###
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = num_hidden_layers),
            nn.LeakyReLU(negative_slope=0.01), # inplace ?
            nn.Linear(in_features = num_hidden_layers, out_features = num_hidden_layers),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(in_features=num_hidden_layers, out_features = num_outputs)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits
    
class NeuralNetwork_optuna(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden_layers):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = 128),
            nn.LeakyReLU(negative_slope=0.04242861410346148), # before: 0.01
            nn.Dropout(0.09382298344626222),
            nn.Linear(in_features = 128, out_features = 64),
            nn.LeakyReLU(negative_slope=0.04242861410346148),
            nn.Dropout(0.21326313772325148),
            nn.Linear(in_features=64, out_features = num_outputs)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits
    
### 3. Early Stopper class ###
class EarlyStopper():
    def __init__(self, drift_patience, plateau_patience, drift_percentage):
        self.drift_patience = drift_patience
        self.plateau_patience = plateau_patience
        self.drift_percentage = drift_percentage
        self.minimum_loss = np.inf
        self.tolerance_break_counter = 0
        self.plateau_counter = 0 # counts how many consecutive losses remain within the range of minimum_loss and minimum_loss*(1 + drift_percentage)

    def stop_early(self, loss):
        if loss < self.minimum_loss:
            # Reset counters
            self.tolerance_break_counter = 0 
            self.plateau_counter = 0

            # Update loss
            self.minimum_loss = loss
            
        elif loss > self.minimum_loss*(1 + self.drift_percentage):
            self.tolerance_break_counter += 1
        
        else:
            self.plateau_counter += 1

        if self.tolerance_break_counter > self.drift_patience or self.plateau_counter > self.plateau_patience:
            return True
        else:
            return False
