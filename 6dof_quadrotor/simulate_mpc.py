from scipy.integrate import odeint
from scipy import signal
#from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sympy as sp
from pid import PID
from plots import *
from linearize import *
import control as ct
import lqr as lqr
import mpc as mpc
import multirotor
import trajectory_handler
from pathlib import Path
from datetime import datetime
from restriction_handler import Restriction
import pandas as pd

### MULTIROTOR PARAMETERS ###
from parameters.octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d, num_rotors, thrust_to_weight

# Simulation trajectory
#trajectory_type = 'circle_xy' # maybe delete

### Create model of multirotor ###
model = multirotor.Multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors, thrust_to_weight)

### SIMULATION PARAMETERS ###
from parameters.simulation_parameters import time_step, T_sample, N, M


# Input and state values at the equilibrium condition
u_eq = [m*g, 0, 0, 0]
omega_eq = model.get_omega_eq()
X_eq = np.zeros(12)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

### Linearization of the model
A, B, C = model.linearize()

tr = trajectory_handler.TrajectoryHandler()

# Parameters
failed_simulations = 0
total_simulations = 0
dataset_id = 1

simulation_metadata = {
    'sim_id': [],
    'trajectory_type': [],
    'disturbed_inputs': [],
    'simulation_time (s)': [],
    'time_sample (s)': [],
    'N': [],
    'M': [],
    'success': [],
    'RMSe': [],
    'execution_time (s)': [],
    'operation': [],
    'failed_rotors': [],
    'ang_speed_percentages (%)': [],
    'min_phi (rad)': [],
    'max_phi (rad)': [],
    'mean_phi (rad)': [],
    'std_phi (rad)': [],
    'min_theta (rad)': [],
    'max_theta (rad)': [],
    'mean_theta (rad)': [],
    'std_theta (rad)': [],
    'min_psi (rad)': [],
    'max_psi (rad)': [],
    'mean_psi (rad)': [],
    'std_psi (rad)': [],
}
dataset_dataframe = pd.DataFrame(simulation_metadata)
dataset_dataframe['success'] = dataset_dataframe['success'].astype(bool)
dataset_dataframe['disturbed_inputs'] = dataset_dataframe['disturbed_inputs'].astype(bool)

# MPC Implementation

def simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights, control_weights, dataset_name, folder_name, disturb_input=False):
    '''
    Executes a control simulation with MPC given the time step, time sample, total simulation time and the desired trajectory.\n
    dataset_name: name of dataset folder the current simulation will belong to\n
    folder_name: name of the folder that will store the simulation results\n
    (obs: the folder hierarchy is dataset_name/folder_name/)
    '''
    t_samples = np.arange(0,T_simulation, T_sample)

    Bw = B @ model.Gama
    MPC = mpc.mpc(M, N, A, Bw, C, time_step, T_sample, output_weights, control_weights, restrictions)
    MPC.initialize_matrices()
    #try:
    x_mpc_rotors, u_rotors, omega_vector, NN_dataset, metadata = MPC.simulate_future_rotors(model, X0, t_samples, trajectory, omega_eq**2, generate_dataset=True, disturb_input=disturb_input)
    #except Exception as error:
    #    x_mpc_rotors, u_rotors, omega_vector, NN_dataset, metadata = None, None, None, None, None
    #    print('exception:', error)

    if x_mpc_rotors is not None:
        save_path = 'simulations/{}/{}/'.format(dataset_name, folder_name)
        #now = datetime.now()
        #current_time = now.strftime("%m_%d_%Hh-%Mm")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plot_states(x_mpc_rotors, t_samples[:np.shape(x_mpc_rotors)[0]], trajectory=trajectory, u_vector=u_rotors, omega_vector=omega_vector, equal_scales=True, legend=['Force/Moment optimization'], save_path=save_path)
        np.savetxt(save_path + "dataset.csv", NN_dataset, delimiter=",")
        return True, metadata
    return False, metadata

def simulate_batch(trajectory_type, args_vector, restrictions_vector, simulate_disturbances):
    # 3. Simulation of POINT trajectories
    global dataset_id
    global total_simulations
    global failed_simulations
    global dataset_dataframe

    total_simulations += len(args_vector) * len(restrictions_vector)
    if simulate_disturbances: total_simulations *= 2

    tr = trajectory_handler.TrajectoryHandler()
    for args in args_vector:
        for restrictions, output_weights, control_weights, restrictions_metadata in restrictions_vector:
            trajectory = tr.generate_trajectory(trajectory_type, args)
            folder_name = f'{trajectory_type}/' + str(dataset_id)

            # Simulation without disturbances
            print(f'Simulation {dataset_id}/{total_simulations}')
            T_simulation = args[-1]
            simulation_success, simulation_metadata = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights, control_weights, dataset_name, folder_name, disturb_input = False)

            if not simulation_success:
                #restrictions_hover = restrictions.copy()
                #restrictions_hover['y_max'][3:5] /= 2
                #restrictions_hover['y_min'] = -restrictions_hover['y_max'][3:5]
                output_weights_hover = np.copy(output_weights)
                output_weights_hover[3:5] *= 4 # Divide delta_y_max by 2 for phi and theta
                simulation_success, simulation_metadata = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights_hover, control_weights, dataset_name, folder_name, disturb_input = False)

            simulation_metadata = {
                'sim_id': dataset_id,
                'trajectory_type': trajectory_type,
                'disturbed_inputs': False,
                'simulation_time (s)': T_simulation,
                'time_sample (s)': T_sample,
                'N': N,
                'M': M,
                'success': simulation_success,
                'RMSe': simulation_metadata['RMSe'] if simulation_success else 'nan',
                'execution_time (s)': simulation_metadata['execution_time'] if simulation_success else 'nan',
                'operation': restrictions_metadata['operation'],
                'failed_rotors': restrictions_metadata['failed_rotors'],
                'ang_speed_percentages (%)': restrictions_metadata['ang_speed_percentages'],
                'min_phi (rad)': simulation_metadata['min_phi'],
                'max_phi (rad)': simulation_metadata['max_phi'],
                'mean_phi (rad)': simulation_metadata['mean_phi'],
                'std_phi (rad)': simulation_metadata['std_phi'],
                'min_theta (rad)': simulation_metadata['min_theta'],
                'max_theta (rad)': simulation_metadata['max_theta'],
                'mean_theta (rad)': simulation_metadata['mean_theta'],
                'std_theta (rad)': simulation_metadata['std_theta'],
                'min_psi (rad)': simulation_metadata['min_psi'],
                'max_psi (rad)': simulation_metadata['max_psi'],
                'mean_psi (rad)': simulation_metadata['mean_psi'],
                'std_psi (rad)': simulation_metadata['std_psi'],
            }
            simulation_metadata = pd.DataFrame([simulation_metadata])

            if not simulation_success: failed_simulations += 1
            dataset_id += 1
            folder_name = f'{trajectory_type}/' + str(dataset_id)
            dataset_dataframe = pd.concat([dataset_dataframe, simulation_metadata])

            # Simulation with disturbances
            print(f'Simulation {dataset_id}/{total_simulations}')
            if simulate_disturbances:
                simulation_success, simulation_metadata = simulate_mpc(X0, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights, control_weights, dataset_name, folder_name, disturb_input = True)
                
                simulation_metadata = {
                    'sim_id': dataset_id,
                    'trajectory_type': trajectory_type,
                    'disturbed_inputs': True,
                    'simulation_time (s)': T_simulation,
                    'time_sample (s)': T_sample,
                    'N': N,
                    'M': M,
                    'success': simulation_success,
                    'RMSe': simulation_metadata['RMSe'] if simulation_success else 'nan',
                    'execution_time (s)': simulation_metadata['execution_time'] if simulation_success else 'nan',
                    'operation': restrictions_metadata['operation'],
                    'failed_rotors': restrictions_metadata['failed_rotors'],
                    'ang_speed_percentages (%)': restrictions_metadata['ang_speed_percentages'],
                    'min_phi (rad)': simulation_metadata['min_phi'],
                    'max_phi (rad)': simulation_metadata['max_phi'],
                    'mean_phi (rad)': simulation_metadata['mean_phi'],
                    'std_phi (rad)': simulation_metadata['std_phi'],
                    'min_theta (rad)': simulation_metadata['min_theta'],
                    'max_theta (rad)': simulation_metadata['max_theta'],
                    'mean_theta (rad)': simulation_metadata['mean_theta'],
                    'std_theta (rad)': simulation_metadata['std_theta'],
                    'min_psi (rad)': simulation_metadata['min_psi'],
                    'max_psi (rad)': simulation_metadata['max_psi'],
                    'mean_psi (rad)': simulation_metadata['mean_psi'],
                    'std_psi (rad)': simulation_metadata['std_psi'],
                }
                simulation_metadata = pd.DataFrame([simulation_metadata])

                if not simulation_success: failed_simulations += 1
                dataset_id += 1
                dataset_dataframe = pd.concat([dataset_dataframe, simulation_metadata])

def generate_dataset(dataset_name = None):
    if dataset_name is None:
        now = datetime.now()
        current_time = now.strftime("%m_%d_%Hh-%Mm")
        dataset_name = current_time
    rst = Restriction(model, T_sample, N, M)
    
    global dataset_dataframe
    
    # 1. Restrictions vector
    restrictions_performance = rst.restrictions_performance()

    # 2. Generation of trajectory batches
    T_simulation_point = 10
    num_points = 15
    points_args = tr.generate_point_trajectories(num_points, T_simulation_point)
    
    # 3. Simulation of trajectory batches
    simulate_batch('point', points_args, restrictions_performance, simulate_disturbances = True )
    
    dataset_dataframe.to_csv(f'{dataset_name}/dataset_metadata.csv', sep=',', index = False)
    
    print(f'Failed simulations: {failed_simulations}/{total_simulations}') # TODO: deixar mais generico (nao so pontos)

try:
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Hh-%Mm")
    dataset_name = current_time
    generate_dataset(dataset_name)
except KeyboardInterrupt:
    save_path = f'simulations/{dataset_name}/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    dataset_dataframe.to_csv(save_path + '/dataset_metadata.csv', sep=',', index = False)
 