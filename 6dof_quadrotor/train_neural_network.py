import numpy as np
from neural_network import NeuralNetwork, NeuralNetwork_optuna, ControlAllocationDataset, EarlyStopper
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
num_epochs = 1000
batch_size = 128
learning_rate = 0.001
num_outputs = 4
num_neurons_hiddenlayers = 128
batches_per_epoch = 200

# Dataset path
datasets_folder = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/'

load_previous_model = False
previous_model_path = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/model_weights_main.pth'

def train_neural_network():
    # 1. Loading datasets and creating dataloaders
    train_dataset = ControlAllocationDataset(datasets_folder + 'train_dataset.csv', num_outputs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    test_dataset = ControlAllocationDataset(datasets_folder + 'validation_dataset.csv', num_outputs)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    ### 2. Building the Neural Network ###
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    ### 3. Training the Neural Network ###
    num_inputs = test_dataset.num_inputs
    model = NeuralNetwork_optuna(num_inputs, num_outputs, num_neurons_hiddenlayers).to(device) # TODO: automatizar 178 e 4

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
    optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.00016413606390770354, weight_decay=0.00010599318964659744)

    # Test forward

    train_losses = []
    val_losses = []
    earlystopper = EarlyStopper(4, 10, 0.05,)

    #for i_batch, batch_sample in enumerate(train_dataloader):
        #if i_batch == 0:
            #print('input', batch_sample['input'])
            #print('output', batch_sample['output'])
            #print('haaa',model(batch_sample['input']))
            #print('haa2', model(train_dataset2.iloc[0:10, 0:178]))

    #print('len train_dataloader', len(train_dataloader['input']), np.shape(train_dataloader['input']))
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i_batch, batch_sample in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(batch_sample['input'])
            loss = criterion(outputs, batch_sample['output'])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_sample['input'].size(0)
        train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(test_dataloader):
                outputs = model(batch_sample['input'])
                loss = criterion(outputs, batch_sample['output'])
                running_loss += loss.item() * batch_sample['input'].size(0)
        val_loss = running_loss / len(test_dataloader.dataset)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}')

        if earlystopper.stop_early(val_loss):
            print('Stopped early to avoid overfitting')
            break


    # TODO: diferenciar test de validation

    torch.save(model.state_dict(), 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/model_weights_optuna.pth')

    fig = plt.figure()
    x = range(1,len(train_losses) + 1)
    plt.plot(x, train_losses)
    plt.plot(x, val_losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train and Test Losses')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.show()


train_neural_network()

    # 
    # 
    # 
    #  outputs = model(x.unsqueeze(1))
    #     loss = criterion(outputs.squeeze(), f(x))
    #     loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()

    #     if epoch % 100 == 0:
    #     print("Epoch {}: Loss = {}".format(epoch, loss.detach().numpy()))

