import numpy as np
import trajectory_handler
import multirotor
from neural_network import NeuralNetwork, NeuralNetwork_optuna
import torch
from scipy.integrate import odeint
from plots import plot_states
import pandas as pd

use_optuna_model = True

### MULTIROTOR PARAMETERS ###
from parameters.quadrotor_parameters import m, g, I_x, I_y, I_z, l, b, d

### Create model of multirotor ###
multirotor_model = multirotor.multirotor(m, g, I_x, I_y, I_z, b, l, d)

num_neurons_hidden_layers = 128 # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#nn_weights_folder = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/'
#weights_file_name = 'model_weights.pth'

### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, num_outputs
q = num_outputs # Number of MPC outputs (x, y z)
num_inputs = 279 # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
T_simulation = 30 # Total simulation time
t_samples = np.arange(0,T_simulation, T_sample)
t_samples_extended = np.arange(0,2*T_simulation, T_sample)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Input and state values at the equilibrium condition
u_eq = [m*g, 0, 0, 0]
omega_eq = multirotor_model.get_omegas(u_eq)
omega_squared_eq = omega_eq**2
print('omega_squared_eq',omega_squared_eq)

# Trajectory
w=2*np.pi*1/30
tr = trajectory_handler.TrajectoryHandler()
#trajectory = tr.line(2, 2, -1, t_samples_extended)
trajectory = tr.circle_xy(w,4,t_samples_extended)
#trajectory = tr.point(0,0,-1,t_samples)

restrictions = { #TODO: CORRIGIR!!!!!!!!!!!!!!!!!!!!!!!
"delta_u_max": np.linalg.pinv(multirotor_model.Gama) @ [10*m*g*T_sample, 0, 0, 0],
"delta_u_min": np.linalg.pinv(multirotor_model.Gama) @ [-10*m*g*T_sample, 0, 0, 0],
"u_max": np.linalg.pinv(multirotor_model.Gama) @ [m*g, 0, 0, 0],
"u_min": np.linalg.pinv(multirotor_model.Gama) @ [-m*g, 0, 0, 0],
"y_max": 100*np.ones(3),
"y_min": -100*np.ones(3)
}

def simulate_neural_network(nn_weights_folder, file_name, t_samples):

    nn_weights_path = nn_weights_folder + file_name

    # Import N used in neural network model
    #N = 0
    #cmd = 'from ' + nn_weights_folder.replace('/', '.') + 'nn_metadata import N'
    #print('cmd=',cmd)
    #exec(cmd)
    #from dataset_canon.canon_N_50_M_20.global_dataset.nn_metadata import N, M

    print('(Check value) N =', N)
    if N == 0:
        print('N was not imported correctly.')
        return
    # 1. Load Neural Network model
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    if use_optuna_model:
        nn_model = NeuralNetwork_optuna(num_inputs, num_outputs, num_neurons_hidden_layers).to(device)
    else:
        nn_model = NeuralNetwork(num_inputs, num_outputs, num_neurons_hidden_layers).to(device)
    nn_model.load_state_dict(torch.load(nn_weights_path, weights_only=True))
    nn_model.eval()

    x_k = X0

    X_vector = [X0]
    u_vector = []
    omega_vector = []
        
    normalization_df = pd.read_csv(nn_weights_folder + 'normalization_data.csv', header = None)

    # Control loop
    for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
        # Mount input tensor to feed NN
        nn_input = np.array([])

        ref_N = trajectory[k:k+N].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
        if np.shape(ref_N)[0] < q*N:
            #print('kpi',N - int(np.shape(ref_N)[0]/q))
            ref_N = np.concatenate((ref_N, np.tile(trajectory[-1].reshape(-1), N - int(np.shape(ref_N)[0]/q))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento

        # Calculating reference values relative to multirotor's current position at instant k
        position_k = np.tile(x_k[9:], N).reshape(-1)
        ref_N_relative = ref_N - position_k

        # Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
        nn_input = np.concatenate((nn_input, x_k[0:9], ref_N_relative), axis = 0)

        # Normalization of the input
        for i_column in range(num_inputs):
            mean = normalization_df.iloc[0, i_column]
            std = normalization_df.iloc[1, i_column]
            nn_input[i_column] = (nn_input[i_column] - mean)/std

        nn_input = nn_input.astype('float32')

        # Get NN output
        delta_omega_squared = nn_model(torch.from_numpy(nn_input)).detach().numpy()

        # De-normalization of the output
        for i_output in range(num_outputs):
            mean = normalization_df.iloc[0, num_inputs + i_output]
            std = normalization_df.iloc[1, num_inputs + i_output]
            delta_omega_squared[i_output] = mean + std*delta_omega_squared[i_output]
 
        # Applying multirotor restrictions
        delta_omega_squared = np.clip(delta_omega_squared, restrictions['u_min'], restrictions['u_max'])
        
        omega_squared = omega_squared_eq + delta_omega_squared

        # Fixing infinitesimal negative values
        omega_squared = np.clip(omega_squared, a_min=0, a_max=None)

        # omega**2 --> u
        #print('omega_squared',omega_squared)
        u_k = multirotor_model.Gama @ (omega_squared)

        f_t_k, t_x_k, t_y_k, t_z_k = u_k # Attention for u_eq (solved the problem)
        t_simulation = np.arange(t_samples[k], t_samples[k+1], time_step)

        # Update plant control (update x_k)
        # x[k+1] = f(x[k], u[k])
        x_k = odeint(multirotor_model.f2, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
        x_k = x_k[-1]

        X_vector.append(x_k)
        u_vector.append(u_k)
        omega_vector.append(np.sqrt(omega_squared))

    return np.array(X_vector), np.array(u_vector), np.array(omega_vector)

nn_weights_folder_mse = 'training_results/25-04-05 - Hover focused dataset N90 M10 - MSELoss/'
if use_optuna_model: weights_file_name_mse = 'model_weights_MSELoss_optuna.pth'
else: weights_file_name_mse = 'model_weights_MSELoss.pth'

#nn_weights_folder_l1 = 'training_results/25-04-05 - Hover focused dataset N90 M10 - L1Loss/'
#weights_file_name_l1 = 'model_weights_L1Loss.pth'

x_mse, u_vector_mse, omega_vector_mse = simulate_neural_network(nn_weights_folder_mse, weights_file_name_mse, t_samples)

#use_optuna_model = False

#x_mse2, u_vector_mse2, omega_vector_mse2 = simulate_neural_network(nn_weights_folder_mse, weights_file_name_mse, t_samples)

plot_states(x_mse, t_samples[:np.shape(x_mse)[0]], trajectory=trajectory[:len(t_samples)], u_vector=u_vector_mse, omega_vector=omega_vector_mse)
#plot_states(x_l1, t_samples[:np.shape(x_l1)[0]], trajectory=trajectory, u_vector=u_vector_l1, omega_vector=omega_vector_l1)

#plot_states(x_mse, t_samples[:np.shape(x_l1)[0]], trajectory=trajectory, u_vector=u_vector_mse, X_lin=x_l1, legend=['MSE', 'MAE'])