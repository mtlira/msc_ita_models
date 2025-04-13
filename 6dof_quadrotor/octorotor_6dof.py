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
from octorotor_parameters import m, g, I_x, I_y, I_z, l, b, d

print('b =',b)
print('d =', d)

time_step = 1e-3 #5e-3 é um bom valor
T_sample = 1e-2 # MP sample time
T_simulation = 30

t = np.arange(0,T_simulation, time_step)
t_samples = np.arange(0,T_simulation, T_sample)
t_samples_extended = np.arange(0,2*T_simulation, T_sample) # Para não ocorrer estagnação da referência em ref[-1] quando se tem menos que q*N pontos restantes

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Equilibrium point (trivial)
X_eq = np.zeros(12)
#X_eq = np.array([0,0,0,0,0,0,0,0,0,1,1,-1])

# f_t está no eixo do corpo

trajectory_type = 'circle_xy'
include_psi = True

# Open-loop Inputs
def u_(t):
    return [1*m*g, 0, 0, 0]

def u_shm(t):
    w = 2*np.pi*(1/T_simulation)*2
    return [m*g - m*w**2*np.cos(w*t), 0, 0, 0]

def u_spiral(t):
    return [1.2*m*g, 5*np.sin(2*np.pi/0.01/T_simulation*t), 5*np.cos(2*np.pi/0.01/T_simulation*t), 0]

def u_torquex(t):
    return [1.2*m*g, 0.004, 0, 0]

# Input to be considered for the open-loop simulation
u_sim = u_shm

# Input value at the equilibrium condition
u_eq = [m*g, 0, 0, 0]


# Find equilibrium points (for this case, it's trivial that eq point is for angles = 0)
#root = fsolve(fn_solve, p.zeros(9))
#print('eq point:',root)

model = multirotor.multirotor(m, g, I_x, I_y, I_z, b, l, d, num_rotors=8)

# deletar #################################3
omega_eq = model.get_omegas(u_eq)
print('omegas_eq',omega_eq)
#############################################

A, B = linearize(model.f_sym, X_eq, u_eq)
C = np.array([[0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1]])

if include_psi:
    C = np.concatenate((C, np.array([[0,0,1,0,0,0,0,0,0,0,0,0]])), axis = 0)

#_, _, x_lin = openloop_sim_linear(A, B, t, X0, X_eq, u_eq, u_sim)

eig_A, _ = np.linalg.eig(np.array(A, dtype='float'))
print('Eigenvalues of A (open-loop):')
print(eig_A)

# LQR
x_max = [
    np.deg2rad(60),
    np.deg2rad(60),
    np.deg2rad(60),
    np.deg2rad(120),
    np.deg2rad(120),
    np.deg2rad(120),
    10/0.8,
    10/0.8,
    10/0.8,
    3,
    3,
    3]

u_max = [
    1*m*g,
    1.8*m*g+0*I_x*(x_max[0])/0.8,
    1.8*m*g+0*I_y*(x_max[1])/1.5,
    1.8*m*g+0*I_z*(x_max[2])/1.5
    ]

########################################################################################
# LQR - tracking

w = 2*np.pi*1/10
tr = trajectory_handler.TrajectoryHandler()

r_tracking = None
if trajectory_type == 'circle_xy':
    r_tracking = tr.circle_xy(w, 5, t_samples, include_psi = include_psi)

if trajectory_type == 'circle_xz':
    r_tracking = tr.circle_xz(w, 5, t_samples, include_psi = include_psi)

if trajectory_type == 'point':
    r_tracking = tr.point(0, 0, 0, t_samples, include_psi = include_psi)

if trajectory_type == 'line':
    r_tracking = tr.line(1, 1, -1, t_samples, 15, include_psi = include_psi)

if trajectory_type == 'helicoidal':
    r_tracking = tr.helicoidal(w,t_samples, include_psi = include_psi)

#r_tracking = tr.point(0, 0, -1, t_samples)
#r_tracking = tr.helicoidal(w,t_samples)
#r_tracking = tr.line(1, 1, -1, t_samples, 15)

#LQR = lqr.LQR(A, B, C, time_step, T_sample)
#LQR.initialize(x_max, u_max)


# # Teste com feedforward de velocidade
# Cspeed = np.array([
#                    [0,0,0,0,0,0,1,0,0,0,0,0],
#                    [0,0,0,0,0,0,0,1,0,0,0,0],
#                    [0,0,0,0,0,0,0,0,1,0,0,0],
#                    [0,0,0,0,0,0,0,0,0,1,0,0],
#                    [0,0,0,0,0,0,0,0,0,0,1,0],
#                    [0,0,0,0,0,0,0,0,0,0,0,1],
#                    ])

# LQR2 = lqr.LQR(A, B, Cspeed, time_step, T_sample)
# LQR2.initialize(x_max, u_max)

    # Nonlinear simulation
#x_lqr_nonlinear, u_lqr_nonlinear = LQR.simulate2(X0, t_samples, r_tracking, model.f2, u_eq)
    # Linear simulation
#x_lqr_linear = LQR.simulate_linear(X0, t_samples, r_tracking)
#x_lqr_linear3, u_linear_discrete3 = LQR.simulate_linear3(X0, t_samples, r_tracking)
#x_lqr_linear2 = LQR.simulate_linear2(X0, t_samples, r_tracking)

# Speed reference
#r_tracking = tr.speed_reference(r_tracking, t_samples)
#x_lqr_nonlinear_speed, u_lqr_nonlinear_speed = LQR2.simulate_speed_reference(X0, t_samples, r_tracking, model.f2, u_eq)
#x_lqr_linear_speed, u_lqr_linear_speed = LQR2.simulate_linear4_speed_reference(X0, t_samples, r_tracking)
#plot_states_speed(x_lqr_nonlinear, t_samples, x_lqr_linear_speed, r_tracking)

#plot_states(x_lqr_nonlinear, t_samples, trajectory=r_tracking, u_vector=u_lqr_nonlinear)
#plot_states(x_lqr_linear, t_samples, x_lqr_linear2, r_tracking, legend=['Discrete(1)', 'Discrete(2)'])
#plot_states(x_lqr_linear2, t_samples, x_lqr_linear3, r_tracking, legend=['Discrete(2)', 'Discrete(3)'])
#plot_states(x_lqr_nonlinear, t_samples, x_lqr_nonlinear, r_tracking)
#fig = plt.figure()
#plt.plot(t_samples[0:-1], u_lqr_nonlinear[:,0])
#plt.plot(t_samples[0:-1], u_lqr_linear_speed[:,0])
#plt.legend(['normal','speed'])
#plt.show()

#X_lqr_nonlinear, u_vector = lqr1.simulate(X0, t, r_tracking, f2, u_eq) # Não linear
#X_lqr_nonlinear2, u_vector2 = lqr2.simulate2(X0, t_samples, r_tracking, f2, u_eq)
#plot_states(x_lqr_nonlinear, t_samples, x_lqr_linear, r_tracking)
#r_tracking = r_circle_xy2
#plot_states(X_lqr_nonlinear2, t_samples, X_lin = None, trajectory=r_tracking)
#plot_inputs(u_vector,t[0:-1])
#plot_delays(X_lqr_nonlinear, r_tracking, t)
#plot_errors(X_lqr_nonlinear, r_tracking, t)

#######################################################################################

# Open-loop simulation (with input = u_sim)
#plot_states(X, t, x_lin)
#plot_states(X, t)


# Closed-loop (PID controller)
#pid = PID(KP,KI,KD,phi_setpoint,time_step)
#X_vector = pid.simulate(t, X0, u_, time_step, f2)
#plot_states(np.array(X_vector),t)

#########################################################################################

# MPC Implementation

N = 100
M = 10
rho = 1


# Clarification: u is actually (u - ueq) and delta_u is (u-ueq)[k] - (u-ueq)[k-1] in this MPC formulation (i.e., u is in reference to u_eq, not 0)
restrictions = {
    #"delta_u_max": 1.5*m*g*time_step*np.ones(4),
    "delta_u_max": np.array([3*m*g*T_sample, 1*m*g*T_sample, 1*m*g*T_sample, 1*m*g*T_sample]),
    "delta_u_min": np.array([-3*m*g*T_sample, -1*m*g*T_sample, -1*m*g*T_sample, -1*m*g*T_sample]),
    "u_max": [m*g, m*g, m*g, m*g],
    "u_min": [-m*g, -m*g, -m*g, -m*g],
    "y_max": 50*np.ones(4) if include_psi else 50*np.ones(3),
    "y_min": -50*np.ones(4) if include_psi else -50*np.ones(3),
}

#teste = np.array([1,2,3])
#print('1/teste=',1/teste)
#print('1/teste^2=',1/(teste**2))

delta_y_max = np.array([1, 1, 0.8, 1.5]) if include_psi else 20*T_sample*np.ones(3)
#delta_y_max = 1e-6*np.ones(3)


output_weights = 1 / (N*delta_y_max**2) # Deve variar a cada passo de simulação?
control_weights = 1 / (M*restrictions['delta_u_max']**2)

#output_weights = [1,1,3] # Deve variar a cada passo de simulação?
#control_weights = [3,1,1,1]

MPC = mpc.mpc(M, N, A, B, C, time_step, T_sample, output_weights, control_weights, restrictions)
MPC.initialize_matrices()
#X_mpc_nonlinear, u_mpc = MPC.simulate(model.f2, X0, t_samples, r_tracking, u_eq)
#X_mpc_nonlinear_future, u_mpc_future = MPC.simulate_future(model.f2, X0, t_samples, r_tracking, u_eq)
#X_mpc_linear, u_mpc_linear = MPC.simulate_linear(X0, t_samples, r_tracking, u_eq)
#plot_states(X_mpc_nonlinear, t_samples[:np.shape(X_mpc_nonlinear)[0]], X_mpc_nonlinear_future, r_tracking, u_mpc, equal_scales=True, legend=['Present Reference', 'Future Reference', 'Trajectory'])
#plot_states(X_mpc_nonlinear, t_samples[:np.shape(X_mpc_nonlinear_future)[0]], X_mpc_nonlinear_future, r_tracking, u_mpc_future, equal_scales=True)
#plot_inputs(u_mpc, t_samples[0:-1])

x_classic, u_classic = MPC.simulate_future(model.f2,X0, t_samples, r_tracking, u_eq)
plot_states(x_classic, t_samples, trajectory=r_tracking[:len(t_samples)], u_vector=u_classic)
print('wrong plot')
# MPC with actuators

omega_max = np.sqrt(2)*omega_eq
# Failure in omega_0
#omega_max[0] = 0*omega_eq[0]
print('Failed omega_0: max value =',omega_max[0])
u_max = omega_max**2 - omega_eq**2

omega_min = np.zeros(8)
u_min = omega_min**2 - omega_eq**2

restrictions2 = {
    #"delta_u_max": 1.5*m*g*time_step*np.ones(4),
    "delta_u_max": np.linalg.pinv(model.Gama) @ [10*m*g*T_sample, 0, 0, 0],
    "delta_u_min": np.linalg.pinv(model.Gama) @ [-10*m*g*T_sample, 0, 0, 0],
    "u_max": u_max,
    "u_min": u_min,
    "y_max": np.array([20, 20, 20, 1.5]) if include_psi else np.array([20, 20, 20]),
    "y_min": np.array([-20, -20, -20, -1.5]) if include_psi else np.array([-20, -20, -20])
}

output_weights2 = 1 / (N*delta_y_max**2) # Deve variar a cada passo de simulação?
control_weights2 = 1 / (M*restrictions2['delta_u_max']**2)

# print(np.sqrt(np.linalg.pinv(model.Gama) @ [2*m*g, 0, 0, 0]))
# print(np.sqrt(np.linalg.pinv(model.Gama) @ [m*g, 0.1*m*g, 0.1*m*g, 0*m*g]))
# print(np.sqrt(np.linalg.pinv(model.Gama) @ [m*g, -0.1*m*g, -0.1*m*g, -0.1*m*g]))
# print(np.sqrt(np.linalg.pinv(model.Gama) @ [1.5*m*g, 0, 0, 0]))

# print(np.linalg.pinv(model.Gama) @ [-m*g, 0, 0, 0])
# print(np.linalg.pinv(model.Gama) @ [m*g, 0, 0, 0])
# print(np.linalg.pinv(model.Gama) @ [m*g, 0.1*m*g, 0.1*m*g, 0.1*m*g])
# print(np.linalg.pinv(model.Gama) @ [m*g, -0.1*m*g, -0.1*m*g, -0.1*m*g])
# print(np.linalg.pinv(model.Gama) @ [1.5*m*g, 0, 0, 0])


Bw = B @ model.Gama
MPC2 = mpc.mpc(M, N, A, Bw, C, time_step, T_sample, output_weights2, control_weights2, restrictions2)
MPC2.initialize_matrices()
x_mpc_rotors, u_rotors, omega_vector, NN_dataset = MPC2.simulate_future_rotors(model, X0, t_samples, r_tracking, omega_eq**2, generate_dataset=False, disturb_input=False)

if x_mpc_rotors is not None:
    #plot_states(X_mpc_nonlinear_future, t_samples[:np.shape(x_mpc_rotors)[0]], x_mpc_rotors, r_tracking, u_rotors, omega_vector, equal_scales=True, legend=['Force/Moment optimization','Angular speed optimization'])
    plot_states(x_mpc_rotors, t_samples[:np.shape(x_mpc_rotors)[0]], trajectory=r_tracking[:len(t_samples)], u_vector=u_rotors, omega_vector=omega_vector, equal_scales=True, legend=['Force/Moment optimization'])

    save_dataset = str(input('Do you wish to save the generated simulation dataset? (y/n): '))
    if save_dataset == 'y':
        now = datetime.now()
        current_time = now.strftime("%m_%d_%Hh-%Mm")
        Path("simulations/{}/{}/".format(trajectory_type, current_time)).mkdir(parents=True, exist_ok=True)
        print('shape NN dataset=', np.shape(NN_dataset))
        np.savetxt("simulations/{}/{}/dataset.csv".format(trajectory_type, current_time), NN_dataset, delimiter=",")