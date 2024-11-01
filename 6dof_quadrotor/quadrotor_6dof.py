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

time_step = 1e-2 #5e-3 é um bom valor
T_simulation = 100
t = np.arange(0,T_simulation, time_step)

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

# Inputs
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
_, _, x_lin = openloop_sim_linear(A, B, t, X0, X_eq, u_eq, u_sim)

print('shape A =',np.shape(A))
print('shape B =',np.shape(B))

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
    1,
    1,
    1]

u_max = [
    1,
    1+0*I_x*(x_max[0])/0.8,
    1+0*I_y*(x_max[1])/1.5,
    1+0*I_z*(x_max[2])/1.5
    ]

Q = np.eye(12)
for i in range(len(Q)):
    Q[i][i] = 1/(x_max[i]**2)

R = np.eye(4)
for i in range(len(R)):
    R[i][i] = 1/(u_max[i]**2)

#Q = np.eye(12)
#R = np.eye(4)

K, S, E = ct.lqr(A, B, Q, R)
print('np.shape(K)',np.shape(K))
print('K=',K)

linear_sys = signal.StateSpace(A - B*K,B,np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
_, _, delta_xout = signal.lsim(linear_sys, 0, t, X0 = X0 - X_eq)
xout = X_eq + delta_xout
#plot_states(xout, t)

#linear_sys2 = ct.StateSpace(A - B*K, B, np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
#ct_out = ct.forced_response(linear_sys2, t, 0, X0 - X_eq, transpose=True)
#xout2 = X_eq + ct_out.states

#print('shape states lsim =',np.shape(xout))
#print('shape states control =',np.shape(xout2))

#plot_states(xout,t,xout2)

# LQR - tracking
########################################################################################
#K_i = 1*np.ones((4,3))
# K_i = np.array([[0, 0, 0.1],
#                 [0, -0.2, 0],
#                 [0.2, 0, 0],
#                 [0, 0, 0]])
K_i = np.array([[0, 0, 0.2],
                [0, -0.3, 0],
                [0.3, 0, 0],
                [0, 0, 0]])
#K_i = np.zeros((4,3))
C = np.array([[0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1]])
lqr = lqr.LQR(K, K_i, C, time_step)

#r_tracking = -1*np.ones((len(t), 3))
#r_tracking = np.zeros((len(t), 3))
#r_tracking = np.array([0*np.ones(len(t)),
#                       0*np.ones(len(t)),
#                        -2*np.ones(len(t))]).transpose()
w = 2*np.pi*1/30
print('teste',np.array([1,2,3])*np.array([3,2,1]))
r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       -1*t]).transpose()

r_circle_xy = np.array([5*np.sin(w*t),
                       (5 - 5*np.cos(w*t)),
                       np.zeros(len(t))]).transpose()

r_circle_xz = np.array([1*np.sin(w*t),
                       np.zeros(len(t)),
                       (1 - 1*np.cos(w*t))]).transpose()

r_shm_z = np.array([np.zeros(len(t)),
                       np.zeros(len(t)),
                       (-5*np.sin(w*t))]).transpose()

r_point = np.array([np.ones(len(t)),
                       np.ones(len(t)),
                       (np.ones(len(t)))]).transpose()

r_tracking = r_helicoidal
X_vector = lqr.simulate(X0, t, r_tracking, f2, u_eq) # Não linear
print('shape xout linear',np.shape(xout), type(xout))
print(xout[0:2])
print('Shape X_vector =',np.shape(X_vector), type(X_vector))
print(X_vector[0:2])
#plot_states(np.array(X_vector), t)

#######################################################################################
# Attempt 1 - Integrator 

print('shape A-BK',np.shape(A-B*K))
print('shape -B*K_i',np.shape(-B*K_i))

A_a = np.concatenate((A - B*K, -B*K_i), axis = 1)
A_temp = np.concatenate((-C, np.zeros((3,3))), axis = 1)
A_a = np.concatenate((A_a, A_temp), axis = 0)

#A_a = np.array([np.array([A - B*K,   -B*K_i]),
#                np.array([-C,        np.zeros((3,3))])])

G = np.concatenate((np.zeros((12,3)), np.eye(3)), axis = 0)

C_a = np.array([
                [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])

#G = np.array([np.zeros((12,3)), np.eye(3)])
print('shape A_a =',np.shape(A_a))
print('shape G =',np.shape(G))

linear_sys_tracking = signal.StateSpace(A_a, G, C_a, np.zeros((3,3)))
X0 = np.concatenate((X0, [0,0,0]), axis=0)
X_eq= np.concatenate((X_eq, [0,0,0]), axis=0)

_,_, x_tracking = signal.lsim(linear_sys_tracking,r_tracking,t, X0 = X0)
xout_tracking = x_tracking
print('shape xout_tracking',np.shape(xout_tracking))
#plot_states(xout, t, xout_tracking)
print('w =', xout_tracking[-1,12:15])
plot_states(X_vector, t, xout_tracking)

#######################################################################################
# T = np.concatenate((A, B), axis = 1)
# T_ = np.concatenate((C, np.zeros((3,4))), axis = 1)
# T = np.concatenate((T,T_), axis = 0)
# print('shape T', np.shape(T))


# Nx_Nu = sp.Matrix(T).pinv()*np.concatenate((np.zeros((12,3)), np.eye(3)),axis = 0)
# print('Nx_Nu',np.shape(Nx_Nu),Nx_Nu)
# Nx = Nx_Nu[0:12,:]
# Nu = Nx_Nu[12:15,:]
# print('Nx',Nx)
# print('Nu',Nu)


# Open-loop simulation
#plot_states(X, t, x_lin)
#plot_states(X, t)

# Closed-loop
pid = PID(KP,KI,KD,phi_setpoint,time_step)
X_vector = pid.simulate(t, X0, u_, time_step, f2)
#plot_states(np.array(X_vector),t)
