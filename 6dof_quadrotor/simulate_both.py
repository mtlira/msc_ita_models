import numpy as np
import trajectory_handler
import multirotor
from neural_network import *
import torch
from scipy.integrate import odeint
from plots import DataAnalyser
import pandas as pd
import restriction_handler
from simulate_mpc import simulate_mpc, wrap_metadata
import time
from pathlib import Path

use_optuna_model = True

### MULTIROTOR PARAMETERS ###
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, num_rotors, thrust_to_weight

### Create model of multirotor ###
multirotor_model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

num_neurons_hidden_layers = 128 # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#nn_weights_folder = 'dataset_canon/canon_N_90_M_10_hover_only/global_dataset/'
#weights_file_name = 'model_weights.pth'

### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, M, gain_scheduling, include_phi_theta_reference, include_psi_reference
q_neuralnetwork = 3 # Number of MPC outputs (x, y z)
num_inputs = 205 - num_rotors # TODO: AUTOMATIZAR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
T_simulation = 15 # Total simulation time
t_samples = np.arange(0,T_simulation, T_sample)
t_samples_extended = np.arange(0,2*T_simulation, T_sample)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Trajectory type
trajectory_type = 'point'
disturb_input = False
use_optuna_model = True

# Input and state values at the equilibrium condition
#u_eq = [m*g, 0, 0, 0]
omega_eq = multirotor_model.get_omega_eq_hover()
omega_squared_eq = omega_eq**2
print('omega_squared_eq',omega_squared_eq)

# Trajectory
w=2*np.pi*1/10
tr = trajectory_handler.TrajectoryHandler()

if trajectory_type == 'circle_xy':
    r_tracking = tr.circle_xy(w, 5, T_simulation,include_psi_reference, include_phi_theta_reference)

if trajectory_type == 'lissajous_xy':
    r_tracking = tr.lissajous_xy(w, 2, T_simulation,include_psi_reference, include_phi_theta_reference)

if trajectory_type == 'circle_xz':
    r_tracking = tr.circle_xz(w, 5, T_simulation, include_psi_reference, include_phi_theta_reference)

if trajectory_type == 'point':
    r_tracking = tr.point(0, 0, 0, T_simulation,include_psi_reference, include_phi_theta_reference)

if trajectory_type == 'line':
    r_tracking = tr.line(1, 1, -1, 20, T_simulation, include_psi_reference, include_phi_theta_reference)

if trajectory_type == 'helicoidal':
    r_tracking = tr.helicoidal(w,T_simulation, include_psi_reference, include_phi_theta_reference)

def simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_id, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
                    gain_scheduling, disturb_input, num_neurons_hidden_layers, use_optuna_model):
    #dataset_id = 1
    #nn_weights_folder = 'training_results/Training dataset v0 - octorotor/'
    #dataset_mother_folder = nn_weights_folder
    #weights_file_name_mse = 'model_weights_octorotor.pth'
    t_samples = np.arange(0, T_simulation, T_sample)
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

    #nn_weights_folder_l1 = 'training_results/25-04-05 - Hover focused dataset N90 M10 - L1Loss/'
    #weights_file_name_l1 = 'model_weights_L1Loss.pth'

    x_nn, u_nn, omega_nn, nn_metadata = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, use_optuna_model=use_optuna_model,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction)
    
    _, mpc_metadata, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights, gain_scheduling,\
                                disturb_input=disturb_input)
    x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]

    simulation_metadata = wrap_metadata(0, trajectory_type, T_simulation, T_sample, N, M, True, mpc_metadata, restriction_metadata, disturbed_inputs=disturb_input)
    if x_nn is not None and x_mpc is not None:
        Path(simulation_save_path).mkdir(parents=True, exist_ok=True)
        for nn_key in list(nn_metadata.keys()):
            if nn_key not in list(simulation_metadata.keys()):
                simulation_metadata[nn_key] = nn_metadata[nn_key]
        dataset_dataframe = pd.DataFrame(simulation_metadata)
        dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',')

rst = restriction_handler.Restriction(multirotor_model, T_sample, N, M)
restriction, output_weights, control_weights, restriction_metadata = rst.restriction('normal')

simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

if __name__ == '__main__':
    dataset_id = 1
    nn_weights_folder = 'training_results/Training dataset v0 - octorotor/'
    dataset_mother_folder = nn_weights_folder
    weights_file_name = 'model_weights_octorotor.pth'
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_id, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, r_tracking, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, gain_scheduling, disturb_input, num_neurons_hidden_layers, use_optuna_model)