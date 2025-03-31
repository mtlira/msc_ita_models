import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


### 1. Dataset class ###

class ControlAllocationDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataframe = pd.read_csv(dataset_path, header = None).astype('float32')
        self.transform = transform
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        num_inputs = np.shape(self.dataframe)[1] - 4 # There are 4 outputs for the quadcopter # TODO: automatizar a verificação de número de saídas (talvez criar um csv de metadados de simulação)
        input = self.dataframe.iloc[idx, 0:num_inputs]
        output = self.dataframe.iloc[idx, num_inputs:]

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