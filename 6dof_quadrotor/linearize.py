import sympy as sp
import numpy as np
from scipy import signal

# Linearization: https://www.youtube.com/watch?v=9jO9q4jZrSI&pp=ygUWbGluZWFyaXplZCBzdGF0ZSBzcGFjZQ%3D%3D
def linearize(f_sym, X_eq, U_eq = None):
    '''
    Linearizes the model state-space function f_sym = x_dot around the equilibrium point defined by X_eq and U_eq, returning the state-space
    matrices A and B.
    '''
    X_symbols, U_symbols, f = f_sym()
    # f is symbolic
    A = []
    for i in range(0,len(f)):
        row = []
        for j in range(0,len(X_symbols)):
            row.append(f[i].diff(X_symbols[j]))
        A.append(row)
    A = sp.Matrix(A)

    B = []
    for i in range(0,len(f)):
        row = []
        for j in range(0,len(U_symbols)):
            row.append(f[i].diff(U_symbols[j]))
        B.append(row)
    B = sp.Matrix(B)
    'phi theta psi p q r u v w'
    A = A.subs([('phi', X_eq[0]),
                ('theta', X_eq[1]),
                ('psi', X_eq[2]),
                ('p', X_eq[3]),
                ('q', X_eq[4]),
                ('r', X_eq[5]),
                ('u', X_eq[6]),
                ('v', X_eq[7]),
                ('w', X_eq[8])])

    #print('A post subs=',A)

    B = B.subs([('f_t', U_eq[0]),
                ('t_x', U_eq[1]),
                ('t_y', U_eq[2]),
                ('t_z', U_eq[3])])
    
    return A,B

def discretize(A, B, C, time_sample):
    '''
    Discretization of State Space matrices A, B and C according to time_sample.
    '''
    p = np.shape(B)[1]
    q = np.shape(C)[0]
    sys_d = signal.cont2discrete((A, B, C, np.zeros((q, p))), time_sample, 'zoh')
    Ad, Bd, Cd, _, _ = sys_d
    return Ad, Bd, Cd    
    
def openloop_sim_linear(A, B, t, X0, X_eq, U_eq, U_sim):
    '''Simulates the linearized model around X0 and U_eq with the given input U_sim over time t, starting from initial condition X0'''
    # U_sim: function U_sim(t)

    linear_sys = signal.StateSpace(A,B,np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))

    # U_l = Delta_U = simulated_U - U_equilibrium
    U_l = [
    np.ones(len(t)),
    np.ones(len(t)),
    np.ones(len(t)),
    np.ones(len(t)),
    ]

    for i in range(0,len(t)):
        U_l[0][i] = (U_sim(t[i])[0] - U_eq[0])
        U_l[1][i] = (U_sim(t[i])[1] - U_eq[1])
        U_l[2][i] = (U_sim(t[i])[2] - U_eq[2])
        U_l[3][i] = (U_sim(t[i])[3] - U_eq[3])

    delta_X_0 = X0 - X_eq

    U_l = np.transpose(U_l)

    tout, yout, xout = signal.lsim(linear_sys, U_l, t, X0 = delta_X_0)
    

    return tout, yout, xout