import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from plots import DataAnalyser
import time
from scipy.integrate import odeint
import pickle

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
        print('Interity check:')
        for i in range(len(self.dataset)):
            if len(self.dataset[i]) != len(self.dataset[0]):
                print(f'corrupted row: i={i}, {len(self.dataset[i])} samples')
        print('\t There are no corrupted data')

        self.normalize()
        self.dataset = self.dataset.astype(np.float32)

    def get_dataset(self):
        return self.dataset

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

class ControlAllocationDataset_Binary_Short(Dataset):
    '''Class to be used if the total dataset is split into multiple CSV files\n
    mother_folder_path: Path of the folder that contains all the CSV files that make up the total dataset'''
    def __init__(self, dataset: np.ndarray, num_outputs):
        self.dataset = dataset
        print('Dataset type:', type(dataset), type(dataset[0]), type(dataset[0][0]))

        self.num_outputs = num_outputs
        self.num_inputs = len(self.dataset[0]) - num_outputs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        input = self.dataset[idx, :self.num_inputs]
        output = self.dataset[idx, self.num_inputs:]
        dict = {'input': torch.from_numpy(input),
                'output': torch.from_numpy(output)}
        return dict

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
    
class NeuralNetwork_optuna0(nn.Module): # First hyperparameter tuning version
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
    
class NeuralNetwork_optuna1(nn.Module): # Second hyperparameter tuning version
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = 1058),
            nn.LeakyReLU(negative_slope=0.010027561298), # before: 0.01
            #nn.Dropout(0.09382298344626222),
            nn.Linear(in_features = 1058, out_features = 1145),
            nn.LeakyReLU(negative_slope=0.010027561298),
            #nn.Dropout(0.21326313772325148),
            nn.Linear(in_features=1145, out_features = num_outputs)
        )

class NeuralNetwork_optuna2(nn.Module): # Third hyperparameter tuning version
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features = num_inputs, out_features = 1276),
            nn.LeakyReLU(negative_slope=0.036692291), # before: 0.01
            #nn.Dropout(0.09382298344626222),
            nn.Linear(in_features = 1276, out_features = 482),
            nn.LeakyReLU(negative_slope=0.036692291),
            #nn.Dropout(0.21326313772325148),
            nn.Linear(in_features = 482, out_features = 77),
            nn.LeakyReLU(negative_slope=0.036692291), # before: 0.01
            nn.Linear(in_features=77, out_features = num_outputs)
        )
        self.optimizer = 'RMSprop'
        self.opt_leaning_rate = 0.0001397094036
        self.l2_lambda = 7.55128976623e-5

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

class NeuralNetworkSimulator(object):
    def __init__(self, model, N, M, num_inputs, num_rotors, q_eff, u_ref, time_step):
        self.model = model
        self.N = N
        self.M = M
        self.num_inputs = num_inputs
        self.num_rotors = num_rotors
        self.q_eff = q_eff # Number of effective reference coordinates for the neural network: 3 (x, y, z)
        self.u_ref = u_ref # input around which the model was linearized (omega_squared_eq)
        self.time_step = time_step

    def simulate_neural_network(self, X0, nn_weights_folder, file_name, t_samples, trajectory, use_optuna_model, num_neurons_hidden_layers, restriction):
        analyser = DataAnalyser()
        nn_weights_path = nn_weights_folder + file_name
        omega_squared_eq = self.u_ref

        # 1. Load Neural Network model
        #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        #print(f"Using {device} device")
        if use_optuna_model:
            nn_model = NeuralNetwork_optuna2(self.num_inputs, num_rotors)
        else:
            nn_model = NeuralNetwork(self.num_inputs, self.num_rotors, num_neurons_hidden_layers)
        nn_model.load_state_dict(torch.load(nn_weights_path, weights_only=True))
        nn_model.eval()

        x_k = X0

        X_vector = [X0]
        u_vector = []
        omega_vector = []
            
        normalization_df = pd.read_csv(nn_weights_folder + 'normalization_data.csv', header = None)

        # Control loop
        execution_time = 0
        waste_time = 0
        start_time = time.perf_counter()

        ## DEBUG (REMOVE LATER) ##
        #min_omega_squared = np.inf
        ## DEBUG (REMOVE LATER) ##

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            # Mount input tensor to feed NN
            nn_input = np.array([])

            ref_N = trajectory[k:k+self.N, 0:3].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
            if np.shape(ref_N)[0] < self.q_eff*self.N:
                #print('kpi',N - int(np.shape(ref_N)[0]/q))
                ref_N = np.concatenate((ref_N, np.tile(trajectory[-1, :3].reshape(-1), self.N - int(np.shape(ref_N)[0]/self.q_eff))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento

            # Calculating reference values relative to multirotor's current position at instant k
            position_k = np.tile(x_k[9:], self.N).reshape(-1)
            ref_N_relative = ref_N - position_k

            # Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
            nn_input = np.concatenate((nn_input, x_k[0:9], ref_N_relative, restriction['u_max'] + omega_squared_eq), axis = 0)

            # Normalization of the input
            #for i_column in range(num_inputs):
            #    mean = normalization_df.iloc[0, i_column]
            #    std = normalization_df.iloc[1, i_column]
            #    nn_input[i_column] = (nn_input[i_column] - mean)/std
            mean = normalization_df.iloc[0, :self.num_inputs]
            std = normalization_df.iloc[1, :self.num_inputs]
            nn_input = np.array((nn_input - mean) / std)

            nn_input = nn_input.astype('float32')

            # Get NN output
            delta_omega_squared = nn_model(torch.from_numpy(nn_input)).detach().numpy()

            # De-normalization of the output
            #for i_output in range(num_outputs):
            #    mean = normalization_df.iloc[0, num_inputs + i_output]
            #    std = normalization_df.iloc[1, num_inputs + i_output]
            #    delta_omega_squared[i_output] = mean + std*delta_omega_squared[i_output]
            mean = normalization_df.iloc[0, self.num_inputs:]
            std = normalization_df.iloc[1, self.num_inputs:]
            delta_omega_squared = mean + std*delta_omega_squared

            ## DEBUG (REMOVE LATER) ##
            #debug_omega_squared = omega_squared_eq + delta_omega_squared
            #if np.min(debug_omega_squared) < min_omega_squared:
            #    min_omega_squared = np.min(debug_omega_squared)
            ## DEBUG (REMOVE LATER) ##

            # Applying multirotor restrictions
            #delta_omega_squared = np.clip(delta_omega_squared, restriction['u_min'], restriction['u_max'])
            # TODO: Add restrição de rate change (ang acceleration)
            
            omega_squared = omega_squared_eq + delta_omega_squared

            # Fixing infinitesimal values out that violate the constraints
            omega_squared = np.clip(omega_squared, a_min=0, a_max=np.clip(restriction['u_max'] + omega_squared_eq, 0, None))

            # omega**2 --> u
            #print('omega_squared',omega_squared)
            u_k = self.model.Gama @ (omega_squared)

            f_t_k, t_x_k, t_y_k, t_z_k = u_k # Attention for u_eq (solved the problem)
            t_simulation = np.arange(t_samples[k], t_samples[k+1], self.time_step)

            # Update plant control (update x_k)
            # x[k+1] = f(x[k], u[k])
            x_k = odeint(self.model.f2, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = x_k[-1]

            if np.linalg.norm(x_k[9:12] - trajectory[k, :3]) > 100: #or np.max(np.abs(x_k[0:2])) > 1.75:
                print('Simulation exploded.')
                print(f'x_{k} =',x_k)

                metadata = {
                'nn_success': False,
                'num_iterations': len(t_samples)-1,    
                'nn_execution_time (s)': execution_time,
                'nn_RMSe': 'nan',
                'nn_min_phi': 'nan',
                'nn_max_phi': 'nan',
                'nn_mean_phi': 'nan',
                'nn_std_phi': 'nan',
                'nn_min_theta': 'nan',
                'nn_max_theta': 'nan',
                'nn_mean_theta': 'nan',
                'nn_std_theta': 'nan',
                'nn_min_psi': 'nan',
                'nn_max_psi': 'nan',
                'nn_mean_psi': 'nan',
                'nn_std_psi': 'nan',
                }
                return None, None, None, metadata

            waste_start_time = time.perf_counter()
            X_vector.append(x_k)
            u_vector.append(u_k)
            omega_vector.append(np.sqrt(omega_squared))
            waste_end_time = time.perf_counter()
            waste_time += waste_end_time - waste_start_time
        
        end_time = time.perf_counter()

        ## DEBUG (REMOVE LATER) ##
        #print('min omega squared',min_omega_squared)

        X_vector = np.array(X_vector)
        RMSe = analyser.RMSe(X_vector[:, 9:], trajectory[:len(X_vector), :3])
        execution_time = (end_time - start_time) - waste_time

        min_phi = np.min(X_vector[:,0])
        max_phi = np.max(X_vector[:,0])
        mean_phi = np.mean(X_vector[:,0])
        std_phi = np.std(X_vector[:,0])

        min_theta = np.min(X_vector[:,1])
        max_theta = np.max(X_vector[:,1])
        mean_theta = np.mean(X_vector[:,1])
        std_theta = np.std(X_vector[:,1])

        min_psi = np.min(X_vector[:,2])
        max_psi = np.max(X_vector[:,2])
        mean_psi = np.mean(X_vector[:,2])
        std_psi = np.std(X_vector[:,2])

        metadata = {
            'nn_success': True,
            'num_iterations': len(t_samples)-1,    
            'nn_execution_time (s)': execution_time,
            'nn_RMSe': RMSe,
            'nn_min_phi': min_phi,
            'nn_max_phi': max_phi,
            'nn_mean_phi': mean_phi,
            'nn_std_phi': std_phi,
            'nn_min_theta': min_theta,
            'nn_max_theta': max_theta,
            'nn_mean_theta': mean_theta,
            'nn_std_theta': std_theta,
            'nn_min_psi': min_psi,
            'nn_max_psi': max_psi,
            'nn_mean_psi': mean_psi,
            'nn_std_psi': std_psi,
        }

        return np.array(X_vector), np.array(u_vector), np.array(omega_vector), metadata



if __name__ == '__main__':
    pass
    #Teste
    #teste = ControlAllocationDataset_Split('teste/', False, num_rotors)
    teste = ControlAllocationDataset_Binary('../Datasets/Training datasets - v1', False, num_rotors)
    print('first sample\n',teste.__getitem__(0))
    print(teste.num_inputs,teste.num_outputs)

