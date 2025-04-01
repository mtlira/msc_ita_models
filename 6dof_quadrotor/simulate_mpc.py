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

### MULTIROTOR PARAMETERS ###
from quadrotor_parameters import m, g, I_x, I_y, I_z, l, b, d

# Simulation trajectory
#trajectory_type = 'circle_xy' # maybe delete

### Create model of multirotor ###
model = multirotor.multirotor(m, g, I_x, I_y, I_z, b, l, d)

time_step = 1e-3 # Simulation time step #5e-3 é um bom valor
T_sample = 5e-2 # MPC sample time
T_simulation = 20 # Total simulation time
t_samples = np.arange(0,T_simulation, T_sample)

# Input and state values at the equilibrium condition
u_eq = [m*g, 0, 0, 0]
omega_eq = model.get_omegas(u_eq)
X_eq = np.zeros(12)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

### Linearization of the model
A, B = linearize(model.f_sym, X_eq, u_eq)
C = np.array([[0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1]])

# eig_A, _ = np.linalg.eig(np.array(A, dtype='float'))
# print('Eigenvalues of A (open-loop):')
# print(eig_A)

# w = 2*np.pi*1/20# maybe delete
tr = trajectory_handler.TrajectoryHandler()

# trajectory = None # maybe delete
# if trajectory_type == 'circle_xy':
#     trajectory = tr.circle_xy(w, 5, t_samples)

# if trajectory_type == 'circle_xz':
#     trajectory = tr.circle_xz(w, 5, t_samples)

# if trajectory_type == 'point':
#     trajectory = tr.point(0, 0, -1, t_samples)

# if trajectory_type == 'line':
#     trajectory = tr.line(1, 1, -1, t_samples, 15)

# if trajectory_type == 'helicoidal':
#     trajectory = tr.helicoidal(w,t_samples)

# MPC Implementation

N = 50
M = 10

def simulate_mpc(time_step, T_sample, T_simulation, trajectory, restrictions, dataset_name, folder_name):
    '''
    Executes a control simulation with MPC given the time step, time sample, total simulation time and the desired trajectory.\n
    dataset_name: name of dataset folder the current simulation will belong to\n
    folder_name: name of the folder that will store the simulation results\n
    (obs: the folder hierarchy is dataset_name/folder_name/)
    '''
    t_samples = np.arange(0,T_simulation, T_sample)

    delta_y_max = 10*T_sample*np.ones(3)

    output_weights2 = 1 / (N*delta_y_max**2) # Deve variar a cada passo de simulação?
    control_weights2 = 1 / (M*restrictions['delta_u_max']**2)

    Bw = B @ model.Gama
    MPC = mpc.mpc(M, N, A, Bw, C, time_step, T_sample, output_weights2, control_weights2, restrictions)
    MPC.initialize_matrices()
    try:
        x_mpc_rotors, u_rotors, omega_vector, NN_dataset = MPC.simulate_future_rotors(model, X0, t_samples, trajectory, omega_eq**2, generate_dataset=True)
    except:
        x_mpc_rotors = None

    if x_mpc_rotors is not None:
        save_path = 'simulations/{}/{}/'.format(dataset_name, folder_name)
        #now = datetime.now()
        #current_time = now.strftime("%m_%d_%Hh-%Mm")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        plot_states(x_mpc_rotors, t_samples[:np.shape(x_mpc_rotors)[0]], trajectory=trajectory, u_vector=u_rotors, omega_vector=omega_vector, equal_scales=True, legend=['Force/Moment optimization'], save_path=save_path)
        print('shape NN dataset=', np.shape(NN_dataset))
        np.savetxt(save_path + "dataset.csv", NN_dataset, delimiter=",")

def generate_dataset():
    #trajectories_array = tr.generate_trajectories_batch()
    trajectories_array = tr.generate_trajectories_batch()

    dataset_id = 1
    now = datetime.now()
    current_time = now.strftime("%m_%d_%Hh-%Mm")
    for trajectory, T_sample, T_simulation in trajectories_array:
        restrictions_vector = [
            {
            "delta_u_max": np.linalg.pinv(model.Gama) @ [10*m*g*T_sample, 0, 0, 0],
            "delta_u_min": np.linalg.pinv(model.Gama) @ [-10*m*g*T_sample, 0, 0, 0],
            "u_max": np.linalg.pinv(model.Gama) @ [m*g, 0, 0, 0],
            "u_min": np.linalg.pinv(model.Gama) @ [m*g, 0, 0, 0],
            "y_max": 100*np.ones(3),
            "y_min": -100*np.ones(3)
            }
        ]

        for restriction in restrictions_vector:
            dataset_name = current_time
            folder_name = str(dataset_id)
            simulate_mpc(time_step, T_sample, T_simulation, trajectory, restriction, dataset_name, folder_name)
            dataset_id += 1

generate_dataset()
