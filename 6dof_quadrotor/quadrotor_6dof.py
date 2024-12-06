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
    1.5*m*g,
    1.8*m*g+0*I_x*(x_max[0])/0.8,
    1.8*m*g+0*I_y*(x_max[1])/1.5,
    1.8*m*g+0*I_z*(x_max[2])/1.5
    ]

Q = np.eye(12)
for i in range(len(Q)):
    Q[i][i] = 1/(x_max[i]**2)

R = np.eye(4)
for i in range(len(R)):
    R[i][i] = 1/(u_max[i]**2)

#Q = np.eye(12)
#R = np.eye(4)

K, _, _ = ct.lqr(A, B, Q, R)

# LQR - state regulation only (no tracking)
linear_sys = signal.StateSpace(A - B*K,B,np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
_, _, delta_xout = signal.lsim(linear_sys, 0, t, X0 = X0 - X_eq)
xout = X_eq + delta_xout

eig_Acl, _ = np.linalg.eig(np.array(A-B*K, dtype='float'))
print('Eigenvalues of A - B*K (closed-loop):')
print(eig_Acl)
#plot_states(xout, t)

#linear_sys2 = ct.StateSpace(A - B*K, B, np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
#ct_out = ct.forced_response(linear_sys2, t, 0, X0 - X_eq, transpose=True)
#xout2 = X_eq + ct_out.states

#print('shape states lsim =',np.shape(xout))
#print('shape states control =',np.shape(xout2))

#plot_states(xout,t,xout2)

########################################################################################
# LQR - tracking
# Attempt 1 - Integrator 
C = np.array([[0,0,0,0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,0,0,0,1]])

A_a = np.concatenate((A, np.zeros((12,3))), axis = 1)
A_a2 = np.concatenate((-C, np.zeros((3,3))), axis = 1)
A_a = np.concatenate((A_a, A_a2), axis = 0)
A_a2 = _

B_a = np.concatenate((B, np.zeros((3,4))),axis = 0)

G = np.concatenate((np.zeros((12,3)), np.eye(3)), axis = 0)

# LQR augmented
x_max_aug = np.concatenate((x_max, np.array([1,1,1])), axis = 0)
Q_aug = np.eye(15)
for i in range(len(Q_aug)):
    Q_aug[i][i] = 1/(x_max_aug[i]**2)
K_a, _, _ = ct.lqr(A_a, B_a, Q_aug, R) # TODO: Talvez tenha que aumentar Q e R
K2 = K_a[:,0:12] # LQR gain K from augmented LQR version
K_i = K_a[:,12:] # Integrator gain

C_a = np.array([
                [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])

w = 2*np.pi*1/20
r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       -1*t]).transpose()

r_circle_xy = np.array([5*np.sin(w*t),
                       (5 - 5*np.cos(w*t)),
                       np.zeros(len(t))]).transpose()

r_circle_xz = np.array([5*np.sin(w*t),
                       np.zeros(len(t)),
                       (5 - 5*np.cos(w*t))]).transpose()

r_shm_z = np.array([np.zeros(len(t)),
                       np.zeros(len(t)),
                       (-5*np.sin(w*t))]).transpose()

r_point = np.array([np.ones(len(t)),
                       np.ones(len(t)),
                       (-10*np.ones(len(t)))]).transpose()

r_line = np.array([t.clip(min=0,max=50),
                    t.clip(min=0,max=50),
                    -t.clip(min=0,max=50)]).transpose()

r_tracking = r_circle_xy

linear_sys_tracking = signal.StateSpace(A_a - np.matmul(B_a,K_a), G, C_a, np.zeros((3,3)))
X0_aug = np.concatenate((X0, [0,0,0]), axis=0)
X_eq= np.concatenate((X_eq, [0,0,0]), axis=0)

eig_Acl_a, _ = np.linalg.eig(np.array(A_a - np.matmul(B_a,K_a), dtype='float'))
print('Eigenvalues of Aa - Ba*Ka (closed-loop):')
print(eig_Acl_a)

_,_, x_tracking = signal.lsim(linear_sys_tracking,r_tracking,t, X0 = X0_aug)
x_lqr_linear = x_tracking

lqr = lqr.LQR(K2, K_i, C, time_step)

X_lqr_nonlinear, u_vector = lqr.simulate(X0, t, r_tracking, f2, u_eq) # Não linear
plot_states(X_lqr_nonlinear, t, x_lqr_linear, r_tracking)
plot_inputs(u_vector,t[0:-1])
#plot_delays(X_lqr_nonlinear, r_tracking, t)
plot_errors(X_lqr_nonlinear, r_tracking, t)

#######################################################################################

# Open-loop simulation (with input = u_sim)
#plot_states(X, t, x_lin)
#plot_states(X, t)

# Closed-loop (PID controller)
#pid = PID(KP,KI,KD,phi_setpoint,time_step)
#X_vector = pid.simulate(t, X0, u_, time_step, f2)
#plot_states(np.array(X_vector),t)
