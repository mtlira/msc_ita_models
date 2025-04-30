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

def simulate_mpc_nn(X0, multirotor_model, N, M, num_inputs, q_neuralnetwork, omega_squared_eq, dataset_id, dataset_mother_folder, weights_file_name, time_step, T_sample, T_simulation, trajectory, trajectory_type, restriction, restriction_metadata, output_weights, control_weights, \
                    gain_scheduling, disturb_input, num_neurons_hidden_layers):
    #dataset_id = 1
    #nn_weights_folder = 'training_results/Training dataset v0 - octorotor/'
    #dataset_mother_folder = nn_weights_folder
    #weights_file_name_mse = 'model_weights_octorotor.pth'
    t_samples = np.arange(0, T_simulation, time_step)
    analyser = DataAnalyser()
    simulation_save_path = f'{dataset_mother_folder}comparative_simulations/{trajectory_type}/{str(dataset_id)}/'

    simulator = NeuralNetworkSimulator(multirotor_model, N, M, num_inputs, num_rotors, q_neuralnetwork, omega_squared_eq, time_step)

    #nn_weights_folder_l1 = 'training_results/25-04-05 - Hover focused dataset N90 M10 - L1Loss/'
    #weights_file_name_l1 = 'model_weights_L1Loss.pth'

    x_nn, u_nn, omega_nn, nn_metadata = simulator.simulate_neural_network(X0, dataset_mother_folder, weights_file_name, t_samples, trajectory, use_optuna_model=True,\
                                num_neurons_hidden_layers=num_neurons_hidden_layers, restriction=restriction)
    
    _, mpc_metadata, simulation_data = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restriction, output_weights, control_weights, gain_scheduling,\
                                disturb_input=False)
    x_mpc, u_mpc, omega_mpc, _ = simulation_data if simulation_data is not None else [None, None, None, None]

    simulation_metadata = wrap_metadata(0, trajectory_type, T_simulation, T_sample, N, M, True, mpc_metadata, restriction_metadata, disturbed_inputs=disturb_input)
    if x_nn is not None and x_mpc is not None:
        Path(simulation_save_path).mkdir(parents=True, exist_ok=True)
        for nn_key in list(nn_metadata.keys()):
            if nn_key not in list(simulation_metadata.keys()):
                simulation_metadata[nn_key] = nn_metadata[nn_key]
        dataset_dataframe = pd.DataFrame(simulation_metadata)
        dataset_dataframe.to_csv(dataset_mother_folder + 'dataset_metadata.csv', sep=',')

        legend = ['Neural Network', 'MPC', 'Trajectory'] if x_mpc is not None else ['Neural Network', 'Trajectory']
        analyser.plot_states(x_nn, t_samples[:np.shape(x_nn)[0]], X_lin=x_mpc, trajectory=trajectory[:len(t_samples)], u_vector=u_nn, omega_vector=omega_nn,\
                            legend=legend, equal_scales=True, save_path=simulation_save_path)



