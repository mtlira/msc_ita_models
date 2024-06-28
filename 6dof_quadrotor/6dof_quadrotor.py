from scipy.integrate import odeint
from scipy import signal
#from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

t = np.arange(0,20, 1e-3)
m = 10
g = 9.80665
I_x = 0.8
I_y = 0.8
I_z = 1

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

def u_(t):
    return [m*g, 0, 0, 0]

u_eq = [m*g, 0, 0, 0]

def f(X,t):
    phi, theta, psi, p, q, r, u, v, w, x, y, z = X
    f_t, t_x, t_y, t_z = u_(0)
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
       w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - v*(np.cos(psi)*np.sin(phi)*np.sin(theta)) + u*(np.cos(psi)*np.cos(theta)),
       v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - w*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)) + u*(np.cos(theta)*np.sin(psi)),
       w*(np.cos(phi)*np.cos(theta)) - u*(np.sin(theta)) + v*(np.cos(theta)*np.sin(phi))
    ]
    return dx_dt

def f_solve(X):
    phi, theta, psi, p, q, r, u, v, w = X[0:9]
    f_t, t_x, t_y, t_z = u_(0)
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
       w*(sp.sin(phi)*sp.sin(psi) + sp.cos(phi)*sp.cos(psi)*sp.sin(theta)) - v*(sp.cos(psi)*sp.sin(phi)*sp.sin(theta)) + u*(sp.cos(psi)*sp.cos(theta)),
       v*(sp.cos(phi)*sp.cos(psi) + sp.sin(phi)*sp.sin(psi)*sp.sin(theta)) - w*(sp.cos(psi)*sp.sin(phi) - sp.cos(phi)*sp.sin(psi)*sp.sin(theta)) + u*(sp.cos(theta)*sp.sin(psi)),
       w*(sp.cos(phi)*sp.cos(theta)) - u*(sp.sin(theta)) + v*(sp.cos(theta)*sp.sin(phi))
    ]
    return X_symbols, U_symbols, dx_dt

X = odeint(f, y0=[0,0,0,0,0,0,0,0,0,0,0,0], t=t)

# Find equilibrium points (for this case, it's trivial that eq point is for angles = 0)
#root = fsolve(f_solve, np.zeros(9))
#print('eq point:',root)

# Linearization: https://www.youtube.com/watch?v=9jO9q4jZrSI&pp=ygUWbGluZWFyaXplZCBzdGF0ZSBzcGFjZQ%3D%3D

def linearize():
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

    return A, B


def plot_states(X,t):
    # Rotation
    fig, axs = plt.subplots(2, 3)
    axs[0,0].plot(t,X[:,0])
    axs[0,0].set_title('$\\phi(t)$')

    axs[0,1].plot(t,X[:,1])
    axs[0,1].set_title('$\\theta(t)$')

    axs[0,2].plot(t,X[:,2])
    axs[0,2].set_title('$\\psi(t)$')

    axs[1,0].plot(t,X[:,3])
    axs[1,0].set_title('p(t)')

    axs[1,1].plot(t,X[:,4])
    axs[1,1].set_title('q(t)')

    axs[1,2].plot(t,X[:,5])
    axs[1,2].set_title('r(t)')

    # Translation
    fig, axs = plt.subplots(2, 3)
    axs[0,0].plot(t,X[:,6])
    axs[0,0].set_title('u(t)')

    axs[0,1].plot(t,X[:,7])
    axs[0,1].set_title('v(t)')

    axs[0,2].plot(t,X[:,8])
    axs[0,2].set_title('w(t)')

    axs[1,0].plot(t,X[:,9])
    axs[1,0].set_title('x(t)')

    axs[1,1].plot(t,X[:,10])
    axs[1,1].set_title('y(t)')

    axs[1,2].plot(t,X[:,11])
    axs[1,2].set_title('z(t)')

    fig = plt.figure()
    axs = plt.axes(projection='3d')
    axs.plot3D(X[:,9], X[:,10], X[:,11]*(-1))
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')

    plt.show()

#plot_states(X,t)
A, B = linearize()

#print('A=',A)
#print('B=',B)

'phi theta psi p q r u v w'
A = A.subs([('phi', 0),
            ('theta', 0),
            ('psi', 0),
            ('p', 0),
            ('q', 0),
            ('r', 0),
            ('u', 0),
            ('v', 0),
            ('w', 0)])

#print('A post subs=',A)

B = B.subs([('f_t', u_eq[0]),
            ('t_x', u_eq[1]),
            ('t_y', u_eq[2]),
            ('t_z', u_eq[3])])


linear_sys = signal.StateSpace(A,B,np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))

#U_l = [u_(0)[0]*np.ones(len(t)),
#       u_(0)[1]*np.ones(len(t)),
#       u_(0)[2]*np.ones(len(t)),
#       u_(0)[3]*np.ones(len(t))]


U_l = [(u_(0)[0] - u_eq[0])*np.ones(len(t)),
       (u_(0)[1] - u_eq[1])*np.ones(len(t)),
       (u_(0)[2] - u_eq[2])*np.ones(len(t)),
       (u_(0)[3] - u_eq[3])*np.ones(len(t))]

U_l = np.transpose(U_l)

tout, yout, xout = signal.lsim(linear_sys, U_l, t, X0 = [0,0,0,0,0,0,0,0,0,0,0,0])

plot_states(xout, t)
#plot_states(X,t)