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

m = 10
g = 9.80665
I_x = 0.8
I_y = 0.8
I_z = 0.8

# PID Controller parameters
KP = 6
KD = 3
KI = 3
phi_setpoint = 0

time_step = 1e-4 #5e-3 é um bom valor
T_sample = 1e-2 # MP sample time
T_simulation = 15

t = np.arange(0,T_simulation, time_step)
t_samples = np.arange(0,T_simulation, T_sample)

# Initial condition
X0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

# Equilibrium point (trivial)
#X_eq = np.zeros(12)
X_eq = np.array([0,0,0,0,0,0,0,0,0,1,1,-1])

# x = [  phi (0)
#        theta (1)
#        psi (2)
#        p (3)
#        q (4)
#        r (5)
#        u (6)
#        v (7)
#        w (8)
#        x (9)
#        y (10)
#        z (11)
#      ]

#u = [
#    ft (0)
#    tx (1)
#    ty (2)
#    tz (3)
#]

# f_t está no eixo do corpo

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

# State-space model functions

# Para testar u como argumento (PID)
def f2(X, t, f_t, t_x, t_y, t_z):
    phi, theta, psi, p, q, r, u, v, w, x, y, z = X
    dx_dt = [
       p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
       q*np.cos(phi) - r*np.sin(phi),
       r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
       (I_y - I_z)/I_x * r*q + t_x/I_x,
       (I_z - I_x)/I_y * p*r + t_y/I_y,
       (I_x - I_y)/I_z * p*q + t_z/I_z,
       r*v - q*w - g*np.sin(theta),
       p*w - r*u + g*np.sin(phi)*np.cos(theta),
       q*u - p*v + g*np.cos(theta)*np.cos(phi) - f_t/m,
       w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - v*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + u*(np.cos(psi)*np.cos(theta)),
       v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - w*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)) + u*(np.cos(theta)*np.sin(psi)),
       w*(np.cos(phi)*np.cos(theta)) - u*(np.sin(theta)) + v*(np.cos(theta)*np.sin(phi))
    ]
    return dx_dt

def f(X,t):
    phi, theta, psi, p, q, r, u, v, w, x, y, z = X
    f_t, t_x, t_y, t_z = u_sim(t)
    dx_dt = [
       p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
       q*np.cos(phi) - r*np.sin(phi),
       r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
       (I_y - I_z)/I_x * r*q + t_x/I_x,
       (I_z - I_x)/I_y * p*r + t_y/I_y,
       (I_x - I_y)/I_z * p*q + t_z/I_z,
       r*v - q*w - g*np.sin(theta),
       p*w - r*u + g*np.sin(phi)*np.cos(theta),
       q*u - p*v + g*np.cos(theta)*np.cos(phi) - f_t/m,
       w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - v*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + u*(np.cos(psi)*np.cos(theta)),
       v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - w*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)) + u*(np.cos(theta)*np.sin(psi)),
       w*(np.cos(phi)*np.cos(theta)) - u*(np.sin(theta)) + v*(np.cos(theta)*np.sin(phi))
    ]
    return dx_dt

# Para achar o ponto de equilíbrio
def f_solve(X):
    phi, theta, psi, p, q, r, u, v, w = X[0:9]
    f_t, t_x, t_y, t_z = u_eq(0)
    dx_dt = [
       p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
       q*np.cos(phi) - r*np.sin(phi),
       r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
       (I_y - I_z)/I_x * r*q + t_x/I_x,
       (I_z - I_x)/I_y * p*r + t_y/I_y,
       (I_x - I_y)/I_z * p*q + t_z/I_z,
       r*v - q*w - g*np.sin(theta),
       p*w - r*u + g*np.sin(phi)*np.cos(theta),
       q*u - p*v + g*np.cos(theta)*np.cos(phi) - f_t/m
    ]
    return dx_dt

# Para linearização
def f_sym():
    X_symbols = sp.symbols('phi theta psi p q r u v w x y z')
    phi, theta, psi, p, q, r, u, v, w, x, y, z = X_symbols
    U_symbols = sp.symbols('f_t t_x t_y t_z')
    f_t, t_x, t_y, t_z = U_symbols
    
    dx_dt = [
       p + r*sp.cos(phi)*sp.tan(theta) + q*sp.sin(phi)*sp.tan(theta),
       q*sp.cos(phi) - r*sp.sin(phi),
       r*sp.cos(phi)/sp.cos(theta) + q*sp.sin(phi)/sp.cos(theta),
       (I_y - I_z)/I_x * r*q + t_x/I_x,
       (I_z - I_x)/I_y * p*r + t_y/I_y,
       (I_x - I_y)/I_z * p*q + t_z/I_z,
       r*v - q*w - g*sp.sin(theta),
       p*w - r*u + g*sp.sin(phi)*sp.cos(theta),
       q*u - p*v + g*sp.cos(theta)*sp.cos(phi) - f_t/m,
       w*(sp.sin(phi)*sp.sin(psi) + sp.cos(phi)*sp.cos(psi)*sp.sin(theta)) - v*(sp.cos(phi)*sp.sin(psi) - sp.cos(psi)*sp.sin(phi)*sp.sin(theta)) + u*(sp.cos(psi)*sp.cos(theta)),
       v*(sp.cos(phi)*sp.cos(psi) + sp.sin(phi)*sp.sin(psi)*sp.sin(theta)) - w*(sp.cos(psi)*sp.sin(phi) - sp.cos(phi)*sp.sin(psi)*sp.sin(theta)) + u*(sp.cos(theta)*sp.sin(psi)),
       w*(sp.cos(phi)*sp.cos(theta)) - u*(sp.sin(theta)) + v*(sp.cos(theta)*sp.sin(phi))
    ]
    return X_symbols, U_symbols, dx_dt

X = odeint(f, y0=X0, t=t)

# Find equilibrium points (for this case, it's trivial that eq point is for angles = 0)
#root = fsolve(fn_solve, p.zeros(9))
#print('eq point:',root)

A,B = linearize(f_sym, X_eq, u_eq)
C = np.array([[0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1]])

_, _, x_lin = openloop_sim_linear(A, B, t, X0, X_eq, u_eq, u_sim)

eig_A, _ = np.linalg.eig(np.array(A, dtype='float'))
print('Eigenvalues of A (open-loop):')
print(eig_A)

# LQR
x_max = [
    np.deg2rad(45),
    np.deg2rad(45),
    np.deg2rad(45),
    np.deg2rad(120),
    np.deg2rad(120),
    np.deg2rad(120),
    5/0.8,
    5/0.8,
    5/0.8,
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

w = 2*np.pi*1/20
r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       -1*t]).transpose()

r_circle_xy = np.array([5*np.sin(w*t),
                       (5 - 5*np.cos(w*t)),
                       np.zeros(len(t))]).transpose()

r_circle_xy2 = np.array([5*np.sin(w*t_samples),
                       (5 - 5*np.cos(w*t_samples)),
                       np.zeros(len(t_samples))]).transpose()

r_circle_xz = np.array([5*np.sin(w*t),
                       np.zeros(len(t)),
                       (5 - 5*np.cos(w*t))]).transpose()

r_shm_z = np.array([np.zeros(len(t)),
                       np.zeros(len(t)),
                       (-5*np.sin(w*t))]).transpose()

r_point = np.array([0*np.ones(len(t)),
                       0*np.ones(len(t)),
                       (-10*np.ones(len(t)))]).transpose()

r_point2 = np.array([0*np.ones(len(t_samples)),
                       0*np.ones(len(t_samples)),
                       (-10*np.ones(len(t_samples)))]).transpose()

r_line = np.array([t.clip(min=0,max=6),
                    t.clip(min=0,max=6),
                    -t.clip(min=0,max=6)]).transpose()

r_line2 = np.array([t_samples.clip(min=0,max=8),
                    t_samples.clip(min=0,max=8),
                    -t_samples.clip(min=0,max=8)]).transpose()

r_explode = np.array([np.zeros(len(t)),
                       np.zeros(len(t)),
                       (-20*np.ones(len(t)))]).transpose()

r_tracking = r_circle_xy2

LQR = lqr.LQR(A, B, C, time_step, T_sample)
LQR.initialize(x_max, u_max)

# Nonlinear simulation
x_lqr_nonlinear, u_lqr_nonlinear = LQR.simulate2(X0, t_samples, r_tracking, f2, u_eq)
#print('shape nonlinear', np.shape(x_lqr_nonlinear))
# Linear simulation
x_lqr_linear = LQR.simulate_linear(X0, t_samples, r_tracking)
#print('shape linear', np.shape(x_lqr_linear))

#X_lqr_nonlinear, u_vector = lqr1.simulate(X0, t, r_tracking, f2, u_eq) # Não linear
#X_lqr_nonlinear2, u_vector2 = lqr2.simulate2(X0, t_samples, r_tracking, f2, u_eq)
plot_states(x_lqr_nonlinear, t_samples, x_lqr_linear, r_tracking)
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

N = 50
M = 5
rho = 1
r_tracking = r_circle_xy2
# 1. Discretization of the space state
# Ad = np.eye(np.shape(A)[0]) + A*time_step
# Bd = B*time_step

# A_tilda = np.concatenate((
#     np.concatenate((A, B), axis = 1),
#     np.concatenate((np.zeros((4,12)), np.eye(4)), axis = 1)
#     ), axis = 0)

# B_tilda = np.concatenate((B, np.eye(4)), axis = 0)

# C_tilda = np.concatenate((C, np.zeros((3,4))), axis = 1)

# phi = C_tilda @ A_tilda
# for i in range(2, N+1):
#     phi = np.concatenate((phi, C_tilda @ np.linalg.matrix_power(A_tilda, i)), axis = 0)
# print('shape phi',np.shape(phi))

# def initialize_G(A_tilda, B_tilda, C_tilda, N, M):
#     # First column
#     column = C_tilda @ B_tilda
#     for i in range(1,N):
#         column = np.concatenate((column, C_tilda @ np.linalg.matrix_power(A_tilda, i) @ B_tilda), axis = 0)

#     # Other columns
#     G = np.array(column)
#     for i in range(1, M):
#         column = np.roll(column, 1, axis = 0)
#         column[0:3,0:4] = np.zeros((3,4))
#         G = np.concatenate((G, column), axis = 1)
#     return G

# G = initialize_G(A_tilda, B_tilda, C_tilda, N, M)
# print('shape G=',np.shape(G))

# Hqp = 2*(np.transpose(G) @ G + rho*np.eye(np.shape(G)[1]))

# Aqp = np.concatenate((
#     np.eye(M),
#     -np.eye(M),
#     np.tril(np.ones((M,M))),
#     -np.tril(np.ones((M,M))),
#     G,
#     -G
# ), axis = 0)

restrictions = {
    #"delta_u_max": 1.5*m*g*time_step*np.ones(4),
    "delta_u_max": np.array([3*m*g*T_sample, 0.1*m*g*T_sample, 0.1*m*g*T_sample, 0.1*m*g*T_sample]),
    "delta_u_min": np.array([-3*m*g*T_sample, -0.1*m*g*T_sample, -0.1*m*g*T_sample, -0.1*m*g*T_sample]),
    "u_max": [m*g, 0.1*m*g, 0.1*m*g, 0.1*m*g],
    "u_min": [-1*m*g, -0.1*m*g, -0.1*m*g, -0.1*m*g],
    "y_max": 50*np.ones(3),
    "y_min": -50*np.ones(3)
}

#teste = np.array([1,2,3])
#print('1/teste=',1/teste)
#print('1/teste^2=',1/(teste**2))

delta_y_max = 10*T_sample*np.ones(3)
#delta_y_max = 1e-6*np.ones(3)


output_weights = 1 / (N*delta_y_max**2) # Deve variar a cada passo de simulação?
control_weights = 1 / (M*restrictions['delta_u_max']**2)

#output_weights = [1,1,3] # Deve variar a cada passo de simulação?
#control_weights = [3,1,1,1]

MPC = mpc.mpc(M, N, rho, A, B, C, time_step, T_sample, output_weights, control_weights, restrictions)
MPC.initialize_matrices()
X_mpc_nonlinear, u_mpc = MPC.simulate(f2, X0, t_samples, r_tracking, u_eq)
X_mpc_linear, u_mpc_linear = MPC.simulate_linear(X0, t_samples, r_tracking, u_eq)
plot_states(X_mpc_nonlinear, t_samples[:np.shape(X_mpc_nonlinear)[0]], X_mpc_linear, r_tracking, u_mpc)
#plot_inputs(u_mpc, t_samples[0:-1])