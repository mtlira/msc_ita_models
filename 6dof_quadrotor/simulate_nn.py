import numpy as np
import trajectory_handler
import multirotor
from neural_network import NeuralNetwork, NeuralNetwork_optuna
import torch
from scipy.integrate import odeint
from plots import DataAnalyser
import pandas as pd
import restriction_handler
from simulate_mpc import simulate_mpc
import time

use_optuna_model = True

### MULTIROTOR PARAMETERS ###
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, num_rotors, thrust_to_weight

### Create model of multirotor ###
multirotor_model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

num_neurons_hidden_layers = 128 # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#nn_weights_folder = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/'
#weights_file_name = 'model_weights.pth'

### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, M
q_neuralnetwork = 3 # Number of MPC outputs (x, y z)
num_inputs = 205 - num_rotors # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
T_simulation = 30 # Total simulation time
t_samples = np.arange(0,T_simulation, T_sample)
t_samples_extended = np.arange(0,2*T_simulation, T_sample)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Input and state values at the equilibrium condition
#u_eq = [m*g, 0, 0, 0]
omega_eq = multirotor_model.get_omega_eq_hover()
omega_squared_eq = omega_eq**2
print('omega_squared_eq',omega_squared_eq)

# Trajectory
w=2*np.pi*1/10
tr = trajectory_handler.TrajectoryHandler()
#trajectory = tr.line(2, 2, -1, 15, T_simulation)
#trajectory = tr.circle_xy(w,5,T_simulation)
trajectory = tr.point(0,0,0,T_simulation)
#trajectory = tr.lissajous_xy(w, 3, T_simulation)

rst = restriction_handler.Restriction(multirotor_model, T_sample, N, M)
restriction, output_weights, control_weights, _ = rst.restriction('total_failure', [0,7])

def simulate_neural_network(nn_weights_folder, file_name, t_samples):
    analyser = DataAnalyser()
    nn_weights_path = nn_weights_folder + file_name

    # 1. Load Neural Network model
    #device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    #print(f"Using {device} device")
    if use_optuna_model:
        nn_model = NeuralNetwork_optuna(num_inputs, num_rotors)
    else:
        nn_model = NeuralNetwork(num_inputs, num_rotors, num_neurons_hidden_layers)
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
    for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
        # Mount input tensor to feed NN
        nn_input = np.array([])

        ref_N = trajectory[k:k+N, 0:3].reshape(-1) # TODO validar se termina em k+N-1 ou em k+N
        if np.shape(ref_N)[0] < q_neuralnetwork*N:
            #print('kpi',N - int(np.shape(ref_N)[0]/q))
            ref_N = np.concatenate((ref_N, np.tile(trajectory[-1, :3].reshape(-1), N - int(np.shape(ref_N)[0]/q_neuralnetwork))), axis = 0) # padding de trajectory[-1] em ref_N quando trajectory[k+N] ultrapassa ultimo elemento

        # Calculating reference values relative to multirotor's current position at instant k
        position_k = np.tile(x_k[9:], N).reshape(-1)
        ref_N_relative = ref_N - position_k

        # Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
        nn_input = np.concatenate((nn_input, x_k[0:9], ref_N_relative, restriction['u_max'] + omega_squared_eq), axis = 0)

        # Normalization of the input
        #for i_column in range(num_inputs):
        #    mean = normalization_df.iloc[0, i_column]
        #    std = normalization_df.iloc[1, i_column]
        #    nn_input[i_column] = (nn_input[i_column] - mean)/std
        mean = normalization_df.iloc[0, :num_inputs]
        std = normalization_df.iloc[1, :num_inputs]
        nn_input = np.array((nn_input - mean) / std)

        nn_input = nn_input.astype('float32')

        # Get NN output
        delta_omega_squared = nn_model(torch.from_numpy(nn_input)).detach().numpy()

        # De-normalization of the output
        #for i_output in range(num_outputs):
        #    mean = normalization_df.iloc[0, num_inputs + i_output]
        #    std = normalization_df.iloc[1, num_inputs + i_output]
        #    delta_omega_squared[i_output] = mean + std*delta_omega_squared[i_output]
        mean = normalization_df.iloc[0, num_inputs:]
        std = normalization_df.iloc[1, num_inputs:]
        delta_omega_squared = mean + std*delta_omega_squared

        # Applying multirotor restrictions
        delta_omega_squared = np.clip(delta_omega_squared, restriction['u_min'], restriction['u_max'])
        # TODO: Add restrição de rate change (ang acceleration)
        
        omega_squared = omega_squared_eq + delta_omega_squared

        # Fixing infinitesimal values out that violate the constraints
        omega_squared = np.clip(omega_squared, a_min=0, a_max=restriction['u_max'] + omega_squared_eq)

        # omega**2 --> u
        #print('omega_squared',omega_squared)
        u_k = multirotor_model.Gama @ (omega_squared)

        f_t_k, t_x_k, t_y_k, t_z_k = u_k # Attention for u_eq (solved the problem)
        t_simulation = np.arange(t_samples[k], t_samples[k+1], time_step)

        # Update plant control (update x_k)
        # x[k+1] = f(x[k], u[k])
        x_k = odeint(multirotor_model.f2, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
        x_k = x_k[-1]

        waste_start_time = time.perf_counter()
        X_vector.append(x_k)
        u_vector.append(u_k)
        omega_vector.append(np.sqrt(omega_squared))
        waste_end_time = time.perf_counter()
        waste_time += waste_end_time - waste_start_time
    
    end_time = time.perf_counter()

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
        'num_iterations': len(t_samples)-1,    
        'nn_execution_time': execution_time,
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

    nn_weights_folder_mse = '../Datasets/Training datasets - v0/'
    weights_file_name_mse = 'model_weights_octorotor.pth'
    analyser = DataAnalyser()

    #nn_weights_folder_l1 = 'training_results/25-04-05 - Hover focused dataset N90 M10 - L1Loss/'
    #weights_file_name_l1 = 'model_weights_L1Loss.pth'

    x_nn, u_nn, omega_nn = simulate_neural_network(nn_weights_folder_mse, weights_file_name_mse, t_samples)
    _, _, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights)
    x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]

    #use_optuna_model = False

    #x_mse2, u_vector_mse2, omega_vector_mse2 = simulate_neural_network(nn_weights_folder_mse, weights_file_name_mse, t_samples)
    legend = ['Neural Network', 'MPC', 'Trajectory'] if x_mpc is not None else ['Neural Network', 'Trajectory']
    analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], X_lin=x_mpc, trajectory=trajectory[:len(t_samples)], u_vector=u_nn, omega_vector=omega_nn, legend=legend, equal_scales=True)
    #plot_states(x_l1, t_samples[:np.shape(x_l1)[0]], trajectory=trajectory, u_vector=u_vector_l1, omega_vector=omega_vector_l1)

    #plot_states(x_mse, t_samples[:np.shape(x_l1)[0]], trajectory=trajectory, u_vector=u_vector_mse, X_lin=x_l1, legend=['MSE