import numpy as np
from neural_network import NeuralNetwork, ControlAllocationDataset, EarlyStopper
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
from torch import nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
num_epochs = 40
batch_size = 128
learning_rate = 0.001
num_outputs = 4
num_neurons_hiddenlayers = 128
batches_per_epoch = 200

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Dataset path
def load_dataset(datasets_folder):
    train_dataset = ControlAllocationDataset(datasets_folder + 'train_dataset.csv', num_outputs)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    validation_dataset = ControlAllocationDataset(datasets_folder + 'validation_dataset.csv', num_outputs)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,num_workers=0)

    num_inputs = validation_dataset.num_inputs

    return train_dataloader, validation_dataloader, num_inputs

# Get the datasets
datasets_folder = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/'

train_dataloader, validation_dataloader, num_inputs = load_dataset(datasets_folder)

def define_model(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    leakyReLU_negative_slope = trial.suggest_float('leakyReLU_negative_slope', 0.0005, 0.1)
    layers = []
    in_features = num_inputs
    for i in range(n_layers):
        out_features = trial.suggest_int('n_units_l{}'.format(i), 64, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(negative_slope=leakyReLU_negative_slope))
        #dropout_percentage = trial.suggest_float('dropout_l{}'.format(i), 0.0, 0.5) # Dropout removed, at least temporarily
        #layers.append(nn.Dropout(dropout_percentage))
        in_features = out_features
    
    layers.append(nn.Linear(in_features, num_outputs))

    return nn.Sequential(*layers)


def objective(trial):
    
    # Generate the model
    num_inputs = validation_dataloader
    model = define_model(trial).to(device)

    # Generate the optimizers
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float('optimizer_learning_rate', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('l2_lambda', 0, 0.5)
    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate, weight_decay = weight_decay)

    # Loss function
    criterion = torch.nn.MSELoss()

    # Model training
    val_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i_batch, batch_sample in enumerate(train_dataloader):
            if i_batch + 1 >= batches_per_epoch:
                break
            optimizer.zero_grad()
            outputs = model(batch_sample['input'])
            loss = criterion(outputs, batch_sample['output'])
            loss.backward()
            optimizer.step()
            #running_loss += loss.item() * batch_sample['input'].size(0)
        #train_loss = running_loss / len(train_dataloader.dataset)
        #train_losses.append(train_loss)

        # Validation loop
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i_batch, batch_sample in enumerate(validation_dataloader):
                # Limiting validation data.
                if i_batch + 1 >= batches_per_epoch:
                    break
                outputs = model(batch_sample['input'])
                loss = criterion(outputs, batch_sample['output'])
                running_loss += loss.item() * batch_sample['input'].size(0)
        val_loss = running_loss / np.min([len(validation_dataloader.dataset), batches_per_epoch*batch_size])
        #val_losses.append(val_loss)

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
    return val_loss

def tune_hyperparameters():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials = 100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('Study statistics:')
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

tune_hyperparameters()
