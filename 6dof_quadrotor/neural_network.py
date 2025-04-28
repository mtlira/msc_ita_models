import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from parameters.octorotor_parameters import num_rotors

### 1. Dataset class ###

class ControlAllocationDataset(Dataset):
    '''Class to be used if the dataset is a single CSV file'''
    def __init__(self, dataset_path, num_outputs, transform=None):
        self.dataframe = pd.read_csv(dataset_path, header = None).astype('float32')
        self.transform = transform
        self.num_outputs = num_rotors
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

class ControlAllocationDataset_Binary(Dataset):
    '''Class to be used if the total dataset is split into multiple CSV files\n
    mother_folder_path: Path of the folder that contains all the CSV files that make up the total dataset'''
    def __init__(self, mother_folder_path, has_header, num_outputs):
        self.normalization_file_name = 'normalization_data.csv'
        self.mother_folder_path = mother_folder_path
        self.has_header = has_header
        self.num_outputs = num_outputs
        # Creating list of all the csv dataset paths
        #self.npy_files = []
        self.dataset = []
        print('Loading dataset indexes...')
        for subdir, _, files in os.walk(self.mother_folder_path):
            for file in files:
                if file == 'dataset.npy':
                    #self.npy_files.append(os.path.join(subdir, file))
                    self.dataset.append(np.load(os.path.join(subdir, file)))
        self.dataset = np.concatenate(self.dataset, axis = 0)
        self.num_inputs = len(self.dataset[0]) - num_outputs
        
        print(f'\tLoaded {len(self.dataset)} samples')
        print(f'\tSample length: {len(self.dataset[0])}')
        print(f'\tDataset size: {self.dataset.nbytes / 1024**2} MB')

        self.normalize()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input = self.dataset[idx, :self.num_inputs]
        output = self.dataset[idx, self.num_inputs:]
        dict = {'input': torch.tensor(input, dtype=torch.float32),
                'output': torch.tensor(output, dtype=torch.float32)}
        return dict
    
    def normalize(self):
        print('Normalizing the Dataset...')
        if not os.path.isfile(self.mother_folder_path + self.normalization_file_name):
            self.mean = np.mean(self.dataset, axis = 0)
            self.std = np.std(self.dataset, axis = 0)
            data = np.concatenate(([self.mean], [self.std]), axis = 0)
            np.savetxt(self.mother_folder_path + 'normalization_data.csv', data, delimiter=",")
        
        else:
            print('\tNormalization dataset file already exists - loading mean and std')
            normalization_df = pd.read_csv(self.mother_folder_path + 'normalization_data.csv', header = None)
            self.mean = np.array([normalization_df.iloc[0, :]])
            self.std = np.array([normalization_df.iloc[1, :]])

        self.dataset = (self.dataset - self.mean) / self.std
        #self.dataset = np.array([(row - self.mean) / self.std for row in self.dataset])

class ControlAllocationDataset_Split(Dataset):
    '''Class to be used if the total dataset is split into multiple CSV files\n
    mother_folder_path: Path of the folder that contains all the CSV files that make up the total dataset'''
    def __init__(self, mother_folder_path, has_header, num_outputs):
        self.normalization_file_name = 'normalization_data.csv'
        self.mother_folder_path = mother_folder_path
        self.has_header = has_header
        self.num_outputs = num_outputs
        # Creating list of all the csv dataset paths
        self.csv_files = []
        print('Loading dataset indexes...')
        for subdir, _, files in os.walk(self.mother_folder_path):
            for file in files:
                if 'dataset.csv' in file and 'metadata' not in file and 'normalization' not in file and '04_21_21h-19m' in subdir: # Skips metadata datasets
                    self.csv_files.append(os.path.join(subdir, file))

        
        self.file_index = [] # Maps global idx to file + local row index
        for file_path in self.csv_files:
            n_rows = sum(1 for _ in open(file_path))
            if has_header: n_rows -= 1
            self.file_index.extend([(file_path, i) for i in range(n_rows)])
        
        self.num_inputs = np.shape(pd.read_csv(self.csv_files[0], skiprows = 0, nrows = 1, header = 0 if has_header else None))[1] - self.num_outputs
        print('\tNumber of samples:', len(self.file_index))
        print('\tNumber of inputs:',self.num_inputs)
        print('\tNumber of outputs',self.num_outputs)
        
        if not os.path.isfile(mother_folder_path + self.normalization_file_name):
            self.normalize()
        
        normalized_data = pd.read_csv(mother_folder_path + self.normalization_file_name, sep = ',', header = 0 if has_header else None)
        self.mean = normalized_data.iloc[0, :].values.squeeze()
        self.std = normalized_data.iloc[1, :].values.squeeze()
    def __len__(self):
        return len(self.file_index)
    
    def __getitem__(self, idx):
        '''Given a global idx (i.e, regarding the global dataset as a whole), get a sample. Performs the translation from the global idx to the individual file and row the sample is located'''
        file_path, row_idx = self.file_index[idx]
        skiprows = row_idx
        if self.has_header:
            skiprows += 1
        # Read just the specific row
        df = pd.read_csv(file_path, skiprows = skiprows, nrows = 1, header = 0 if self.has_header else None)
        data = df.values.squeeze() # Converts DataFrame to array

        input_normalized = (data[:self.num_inputs] - self.mean[:self.num_inputs]) / self.std[:self.num_inputs]
        output_normalized = (data[self.num_inputs:] - self.mean[self.num_inputs:]) / self.std[self.num_inputs:]

        sample = {
            'input': torch.tensor(input_normalized, dtype=torch.float32),
            'output': torch.tensor(output_normalized, dtype=torch.float32)
            }
        return sample
    
    def normalize(self):
        print('Normalizing the Dataset')
        sum = np.zeros(self.num_inputs + self.num_outputs)
        sum_squared = np.zeros(self.num_inputs + self.num_outputs)
        num_samples = 0
        delta_squared = 0

        # Mean
        for file_path in self.csv_files:
            df = pd.read_csv(file_path, header = 0 if self.has_header else None)
            num_samples += len(df)
            sum += df.sum(axis = 0)
        
        mean = (sum / num_samples)

        # Std
        for file_path in self.csv_files:
            df = pd.read_csv(file_path, header = 0 if self.has_header else None)
            delta_squared += ((df - mean)**2).sum(axis = 0)
            #sum_squared += (df**2).sum()
        
        std = np.sqrt(delta_squared / (num_samples-1))
        #std = np.sqrt(sum_squared/num_samples - mean**2)
        norm_data = pd.concat([mean, std], ignore_index = True, axis = 1).T
        norm_data.to_csv(self.mother_folder_path + self.normalization_file_name, header = True if self.has_header else False, index = False)

        return mean, std


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
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = 128),
            nn.LeakyReLU(negative_slope=0.04242861410346148), # before: 0.01
            #nn.Dropout(0.09382298344626222),
            nn.Linear(in_features = 128, out_features = 64),
            nn.LeakyReLU(negative_slope=0.04242861410346148),
            #nn.Dropout(0.21326313772325148),
            nn.Linear(in_features=64, out_features = num_outputs)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits
    
### 3. Early Stopper class ###
class EarlyStopper():
    def __init__(self, drift_patience, plateau_patience, drift_percentage, save_path):
        self.drift_patience = drift_patience
        self.plateau_patience = plateau_patience
        self.drift_percentage = drift_percentage
        self.minimum_loss = np.inf
        self.tolerance_break_counter = 0
        self.plateau_counter = 0 # counts how many consecutive losses remain within the range of minimum_loss and minimum_loss*(1 + drift_percentage)
        self.save_path = save_path

    def stop_early(self, loss, model):
        if loss < self.minimum_loss:
            # Reset counters
            self.tolerance_break_counter = 0 
            self.plateau_counter = 0

            # Update loss
            self.minimum_loss = loss

            # Save model's weights in save_path
            torch.save(model.state_dict(), self.save_path + 'model_weights.pth')
            
        elif loss > self.minimum_loss*(1 + self.drift_percentage):
            self.tolerance_break_counter += 1
        
        else:
            self.plateau_counter += 1

        if self.tolerance_break_counter > self.drift_patience or self.plateau_counter > self.plateau_patience:
            return True
        else:
            return False


if __name__ == '__main__':
    pass
    #Teste
    #teste = ControlAllocationDataset_Split('teste/', False, num_rotors)
    teste = ControlAllocationDataset_Binary('simulations/04_27_00h-26m/', False, num_rotors)
    print('first sample\n',teste.__getitem__(0))
    print(teste.num_inputs,teste.num_outputs)

