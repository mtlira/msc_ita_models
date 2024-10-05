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

m = 10
g = 9.80665
I_x = 0.8
I_y = 0.8
I_z = 0.8

# Controller parameters
KP = 6
KD = 3
KI = 3
phi_setpoint = 0
time_step = 1e-3
T_simulation = 10
t = np.arange(0,T_simulation, time_step)

# Initial condition
X0 = np.array([0.2,0,0,0,0,0,0,0,0,0,0,0])

# Equilibrium point (trivial)
#X_eq = np.zeros(12)
X_eq = np.array([0,0,0,0,0,0,0,0,0,0,0,-10])

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

# Para achar o ponto de equilíbario
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
#root = fsolve(f_solve, np.zeros(9))
#print('eq point:',root)


A,B = linearize(f_sym, X_eq, u_eq)
_, _, x_lin = openloop_sim_linear(A, B, t, X0, X_eq, u_eq, u_sim)

print('shape A =',np.shape(A))
print('shape B =',np.shape(B))

# LQR

Q = np.eye(12)
R = 2*np.eye(4)

K, S, E = ct.lqr(A, B, Q, R)

linear_sys = signal.StateSpace(A - B*K,B,np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
_, _, delta_xout = signal.lsim(linear_sys, 0, t, X0 = X0 - X_eq)
xout = X_eq + delta_xout
#plot_states(xout, t)

linear_sys2 = ct.StateSpace(A - B*K, B, np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
ct_out = ct.forced_response(linear_sys2, t, 0, X0 - X_eq, transpose=True)
xout2 = X_eq + ct_out.states

print('shape states lsim =',np.shape(xout))
print('shape states control =',np.shape(xout2))

plot_states(xout,t,xout2)

# Open-loop simulation
#plot_states(X, t, x_lin)
#plot_states(X, t)

# Closed-loop
pid = PID(KP,KI,KD,phi_setpoint,time_step)
X_vector = pid.simulate(t, X0, u_, time_step, f2)
#plot_states(np.array(X_vector),t)
