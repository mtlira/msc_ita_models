import numpy as np
from neural_network import NeuralNetwork, ControlAllocationDataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

def train_neural_network():
    train_dataset = ControlAllocationDataset('dataset_canon/canon_N_50_M_20/global/train_dataset.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    test_dataset = ControlAllocationDataset('dataset_canon/canon_N_50_M_20/global/test_dataset.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    ### 2. Building the Neural Network ###
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    ### 3. Training the Neural Network ###
    model = NeuralNetwork(178, 4, 128).to(device) # TODO: automatizar 178 e 4

    # for i_batch, batch_sample in enumerate(dataloader):
    #     if i_batch == 0:
    #         print('batch sample[input] shape', np.shape(batch_sample['input']))
    #         print('batch sample[output] shape', np.shape(batch_sample['output']))
    #         print('model\n',model)
    #         print('forward\n',model(batch_sample['input']), '\n shape',model(batch_sample['input']).size())

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Using Adam

    # Test forward

    train_losses = []
    val_losses = []

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

    # TODO: diferenciar test de validation

    fig = plt.figure()
    x = range(1,num_epochs + 1)
    plt.plot(x, train_losses)
    plt.plot(x, val_losses)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title('Train and Test Losses')
    plt.legend(['Train Loss', 'Test Loss'])
    plt.show()

    torch.save(model.state_dict(), 'model_weights.pth')

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

