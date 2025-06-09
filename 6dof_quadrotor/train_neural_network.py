import numpy as np
from neural_network import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
import pickle

from parameters.octorotor_parameters import num_rotors


def train_neural_network():
    # Hyperparameters
    num_epochs = 1000
    batch_size = 128
    #learning_rate = 0
    num_outputs = num_rotors
    #num_neurons_hiddenlayers = 128
    optuna_version = 'v4'
    # Dataset path
    datasets_folder = f'../Datasets/Training datasets - {optuna_version}/'

    print('CUDA:', torch.cuda.is_available(), torch.accelerator.is_available())
    print(torch.cuda.is_available())  # Should print True
    print(torch.cuda.device_count())  # Should be > 0
    print(torch.cuda.get_device_name(0)) 


    load_previous_model = False
    previous_model_path = ''

    training_metadata = {
        'epoch': [],
        'train_loss': [],
        'validation_loss': [],
        'execution_time': []
    }

    # 1. Loading datasets and creating dataloaders
    global_dataset_preload = ControlAllocationDataset_Binary(datasets_folder, False, num_outputs)
    global_dataset = ControlAllocationDataset_Binary_Short(global_dataset_preload.get_dataset(), num_outputs)
    train_size = int(0.8 * len(global_dataset))
    val_size = int(0.1 * len(global_dataset))
    test_size = len(global_dataset) - train_size - val_size

    #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    print(f"Using {device} device")

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(global_dataset, [train_size, val_size, test_size])

    #train_dataset = ControlAllocationDataset(datasets_folder + 'train_dataset.csv', num_outputs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)

    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)
    
    #test_dataset = ControlAllocationDataset(datasets_folder + 'validation_dataset.csv', num_outputs)
    #test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=0, pin_memory=True if device == 'cuda' else False)


    ### 2. Building the Neural Network ###
    num_inputs = global_dataset.num_inputs
    if optuna_version == 'v2':
        model = NeuralNetwork_optuna2(num_inputs, num_outputs).to(device) # TODO: automatizar 178 e 4
    if optuna_version == 'v3':
        model = NeuralNetwork_optuna3(num_inputs, num_outputs).to(device) # TODO: automatizar 178 e 4
    if optuna_version == 'v4':
        model = NeuralNetwork_optuna4(num_inputs, num_outputs).to(device) # TODO: automatizar 178 e 4
    else:
        raise Exception("Fix code to consider other optuna versions.")

    if load_previous_model:
        model.load_state_dict(torch.load(previous_model_path, weights_only=True))
        model.eval()


    # for i_batch, batch_sample in enumerate(dataloader):
    #     if i_batch == 0:
    #         print('batch sample[input] shape', np.shape(batch_sample['input']))
    #         print('batch sample[output] shape', np.shape(batch_sample['output']))
    #         print('model\n',model)
    #         print('forward\n',model(batch_sample['input']), '\n shape',model(batch_sample['input']).size())

    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Using Adam
    if model.optimizer == "Adam": optimizer = optimizer = torch.optim.Adam(model.parameters(), lr = model.opt_leaning_rate, weight_decay=model.l2_lambda)
    if model.optimizer == "RMSprop": optimizer = torch.optim.RMSprop(model.parameters(), lr = model.opt_leaning_rate, weight_decay=model.l2_lambda)
    if model.optimizer == "SGD": optimizer = torch.optim.SGD(model.parameters(), lr = model.opt_leaning_rate, weight_decay=model.l2_lambda)

    # Test forward
    earlystopper = EarlyStopper(4, 15, 0.05, datasets_folder)

    #for i_batch, batch_sample in enumerate(train_dataloader):
        #if i_batch == 0:
            #print('input', batch_sample['input'])
            #print('output', batch_sample['output'])
            #print('haaa',model(batch_sample['input']))
            #print('haa2', model(train_dataset2.iloc[0:10, 0:178]))

    #print('len train_dataloader', len(train_dataloader['input']), np.shape(train_dataloader['input']))

    ### 3. Training the Neural Network ###

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for i_batch, batch_sample in enumerate(train_dataloader):
            input = batch_sample['input'].to(device, non_blocking = True)
            output = batch_sample['output'].to(device, non_blocking = True)
            #print("input device:", input.device)
            #print("model param device:", next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input.size(0)
        train_loss = running_loss / len(train_dataloader.dataset)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(validation_dataloader):
                input = batch_sample['input'].to(device, non_blocking = True)
                output = batch_sample['output'].to(device, non_blocking = True)
                outputs = model(input)
                loss = criterion(outputs, output)
                running_loss += loss.item() * batch_sample['input'].size(0)
        val_loss = running_loss / len(validation_dataloader.dataset)

        end_time = time.time()

        training_metadata['execution_time'].append(end_time - start_time)
        training_metadata['epoch'].append(epoch + 1)
        training_metadata['train_loss'].append(train_loss)
        training_metadata['validation_loss'].append(val_loss)
        print(f'Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}')


        if earlystopper.stop_early(val_loss, model):
            print('Stopped early to avoid overfitting')
            break

    # Test loss
    if optuna_version == 'v2':
        model = NeuralNetwork_optuna2(num_inputs, num_rotors).to(device)
    if optuna_version == 'v3':
        model = NeuralNetwork_optuna3(num_inputs, num_rotors).to(device)
    if optuna_version == 'v4':
        model = NeuralNetwork_optuna4(num_inputs, num_rotors).to(device)
        model.load_state_dict(torch.load(datasets_folder + 'model_weights.pth', weights_only=True))
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(test_dataloader):
                input = batch_sample['input'].to(device, non_blocking = True)
                output = batch_sample['output'].to(device, non_blocking = True)
                outputs = model(input)
                loss = criterion(outputs, output)
                test_loss += loss.item() * batch_sample['input'].size(0)            
        test_loss = test_loss / len(test_dataloader.dataset)
        with open(datasets_folder + "test_losses.txt", "w") as f:
            f.write(str(test_loss))
    else: print('Test loss not saved. mismatch of optuna versions!!!')

    trim_idx = np.min([len(training_metadata['epoch']), len(training_metadata['train_loss']), len(training_metadata['validation_loss'])])
    training_dataframe = pd.DataFrame(training_metadata)
    training_dataframe.to_csv(datasets_folder + 'training_metadata.csv', sep = ',', index=False)

    fig = plt.figure()
    x = training_metadata['epoch']
    plt.plot(x, training_metadata['train_loss'][:trim_idx])
    plt.plot(x, training_metadata['validation_loss'][:trim_idx])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train and Test Losses')
    plt.legend(['Train Loss', 'Validation Loss'])
    plt.savefig(datasets_folder + 'losses.png')
    plt.savefig(datasets_folder + 'losses.pdf')
    plt.show()

if __name__ == '__main__':
    #try:
    train_neural_network()

    # 
    # 
    #  outputs = model(x.unsqueeze(1))
    #     loss = criterion(outputs.squeeze(), f(x))
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()

    #     if epoch % 100 == 0:
    #     print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

