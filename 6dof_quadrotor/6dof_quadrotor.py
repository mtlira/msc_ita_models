from scipy.integrate import odeint
from scipy import signal
#from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sympy as sp
from pid import PID

#params = {'legend.fontsize': 'large',
#         'axes.labelsize': 'x-large',
#         'axes.titlesize':'x-large',
#        'xtick.labelsize':'large',
#         'ytick.labelsize':'large'}
#pylab.rcParams.update(params)

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
X0 = [0,0,0,0,0,0,0,0,0,0,0,-10]

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
    return [1*m*g, 0, 0, 0]

def u_shm(t):
    w = 2*np.pi*(1/T_simulation)*2
    return [m*g - m*w**2*np.cos(w*t), 0, 0, 0]

def u_spiral(t):
    return [1.2*m*g, 5*np.sin(2*np.pi/0.01/T_simulation*t), 5*np.sin(2*np.pi/0.02/T_simulation*t), 0]

def u_torquex(t):
    return [1.2*m*g, 0.004, 0, 0]

u_sim = u_shm

u_eq = [m*g, 0, 0, 0]

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
       w*(sp.sin(phi)*sp.sin(psi) + sp.cos(phi)*sp.cos(psi)*sp.sin(theta)) - v*(sp.cos(phi)*sp.sin(psi) - sp.cos(psi)*sp.sin(phi)*sp.sin(theta)) + u*(sp.cos(psi)*sp.cos(theta)),
       v*(sp.cos(phi)*sp.cos(psi) + sp.sin(phi)*sp.sin(psi)*sp.sin(theta)) - w*(sp.cos(psi)*sp.sin(phi) - sp.cos(phi)*sp.sin(psi)*sp.sin(theta)) + u*(sp.cos(theta)*sp.sin(psi)),
       w*(sp.cos(phi)*sp.cos(theta)) - u*(sp.sin(theta)) + v*(sp.cos(theta)*sp.sin(phi))
    ]
    return X_symbols, U_symbols, dx_dt

X = odeint(f, y0=X0, t=t)

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


def quali_shm(X,t):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t,(-1)*X[:,11])
    ax.set_title('z(t)')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (m)')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('3D Plot')
    plt.show()

def quali_pid(X,t):
    #PID control

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t,X[:,0])
    ax.set_title('$\\phi(t)$')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('$\\phi (rad)$')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('3D Plot')
    plt.show()

def quali_torquex(X,t):
    fig, axs = plt.subplots(ncols=2, nrows=2)
    gs = axs[1, 1].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1],projection='3d')

    axs[0,0].plot(t,(-1)*X[:,10])
    axs[0,0].set_title('y(t)')
    axs[0,0].set_xlabel('t (s)')
    axs[0,0].set_ylabel('y (m)')

    axs[1,0].plot(t,(-1)*X[:,11])
    axs[1,0].set_title('z(t)')
    axs[1,0].set_xlabel('t (s)')
    axs[1,0].set_ylabel('z (m)')

    axbig.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    axbig.set_xlabel('x (m)')
    axbig.set_ylabel('y (m)')
    axbig.set_zlabel('z (m)')
    axbig.set_title('3D Plot')
    plt.show()

def quali_linear(X,t,X_lin):
    fig, axs = plt.subplots(ncols=2, nrows=2)
    gs = axs[1, 1].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1],projection='3d')

    axs[0,0].plot(t,(-1)*X[:,10])
    axs[0,0].plot(t,(-1)*X_lin[:,10])
    axs[0,0].set_title('y(t)')
    axs[0,0].set_xlabel('t (s)')
    axs[0,0].set_ylabel('y (m)')

    axs[1,0].plot(t,(-1)*X[:,11])
    axs[1,0].plot(t,(-1)*X_lin[:,11])
    axs[1,0].set_title('z(t)')
    axs[1,0].set_xlabel('t (s)')
    axs[1,0].set_ylabel('z (m)')

    axbig.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    axbig.plot3D(X_lin[:,9], X_lin[:,10]*(-1), X_lin[:,11]*(-1))
    axbig.set_xlabel('x (m)')
    axbig.set_ylabel('y (m)')
    axbig.set_zlabel('z (m)')
    axbig.set_title('3D Plot')

    fig.legend(['Non-linear','Linear'])
    plt.show()


def plot_states(X,t, X_lin = None):
    # Rotation
    fig, axs = plt.subplots(2, 3)
    axs[0,0].plot(t,X[:,0])
    if X_lin is not None: axs[0,0].plot(t,X_lin[:,0])
    axs[0,0].set_title('$\\phi(t)$')
    axs[0,0].set_xlabel('t (s)')
    axs[0,0].set_ylabel('$\\phi (rad)$')

    axs[0,1].plot(t,X[:,1])
    if X_lin is not None: axs[0,1].plot(t,X_lin[:,1])
    axs[0,1].set_title('$\\theta(t)$')
    axs[0,1].set_xlabel('t (s)')
    axs[0,1].set_ylabel('$\\theta (rad)$')

    axs[0,2].plot(t,X[:,2])
    if X_lin is not None: axs[0,2].plot(t,X_lin[:,2])
    axs[0,2].set_title('$\\psi(t)$')
    axs[0,2].set_xlabel('t (s)')
    axs[0,2].set_ylabel('$\\psi$ (rad)')

    axs[1,0].plot(t,X[:,3])
    if X_lin is not None: axs[1,0].plot(t,X_lin[:,3])
    axs[1,0].set_title('p(t)')
    axs[1,0].set_xlabel('t (s)')
    axs[1,0].set_ylabel('p (rad/s)')

    axs[1,1].plot(t,(-1)*X[:,4])
    if X_lin is not None: axs[1,1].plot(t,(-1)*X_lin[:,4])
    axs[1,1].set_title('q(t)')
    axs[1,1].set_xlabel('t (s)')
    axs[1,1].set_ylabel('q (rad/s)')

    axs[1,2].plot(t,(-1)*X[:,5])
    if X_lin is not None: axs[1,2].plot(t,(-1)*X_lin[:,5])
    axs[1,2].set_title('r(t)')
    axs[1,2].set_xlabel('t (s)')
    axs[1,2].set_ylabel('r (rad/s)')
    if X_lin is not None: fig.legend(['Non-linear','Linear'])
    plt.subplots_adjust(left=0.083, bottom=0.083, right=0.948, top=0.914, wspace=0.23, hspace=0.31)

    # Translation
    fig, axs = plt.subplots(2, 3)
    axs[0,0].plot(t,X[:,6])
    if X_lin is not None: axs[0,0].plot(t,X_lin[:,6])
    axs[0,0].set_title('u(t)')
    axs[0,0].set_xlabel('t (s)')
    axs[0,0].set_ylabel('u (m/s)')

    axs[0,1].plot(t,(-1)*X[:,7])
    if X_lin is not None: axs[0,1].plot(t,(-1)*X_lin[:,7])
    axs[0,1].set_title('v(t)')
    axs[0,1].set_xlabel('t (s)')
    axs[0,1].set_ylabel('v (m/s)')

    axs[0,2].plot(t,(-1)*X[:,8])
    if X_lin is not None: axs[0,2].plot(t,(-1)*X_lin[:,8])
    axs[0,2].set_title('w(t)')
    axs[0,2].set_xlabel('t (s)')
    axs[0,2].set_ylabel('w (m/s)')

    axs[1,0].plot(t,X[:,9])
    if X_lin is not None: axs[1,0].plot(t,X_lin[:,9])
    axs[1,0].set_title('x(t)')
    axs[1,0].set_xlabel('t (s)')
    axs[1,0].set_ylabel('x (m)')

    axs[1,1].plot(t,(-1)*X[:,10])
    if X_lin is not None: axs[1,1].plot(t,(-1)*X_lin[:,10])
    axs[1,1].set_title('y(t)')
    axs[1,1].set_xlabel('t (s)')
    axs[1,1].set_ylabel('y (m)')

    axs[1,2].plot(t,(-1)*X[:,11])
    if X_lin is not None: axs[1,2].plot(t,(-1)*X_lin[:,11])
    axs[1,2].set_title('z(t)')
    axs[1,2].set_xlabel('t (s)')
    axs[1,2].set_ylabel('z (m)')

    if X_lin is not None: fig.legend(['Non-linear','Linear'])
    plt.subplots_adjust(left=0.083, bottom=0.083, right=0.948, top=0.914, wspace=0.23, hspace=0.31)

    fig = plt.figure()
    axs = plt.axes(projection='3d')
    axs.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    if X_lin is not None: axs.plot3D(X_lin[:,9], X_lin[:,10]*(-1), X_lin[:,11]*(-1))
    axs.set_xlabel('x (m)')
    axs.set_ylabel('y (m)')
    axs.set_zlabel('z (m)')
    axs.set_title('3D Plot')
    if X_lin is not None: fig.legend(['Non-linear','Linear'])
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

tout, yout, xout = signal.lsim(linear_sys, U_l, t, X0 = X0)


# Open-loop simulation
#plot_states(X, t, xout)
#plot_states(X, t)
#quali_linear(X,t,xout)
#quali_torquex(X,t)
quali_shm(X,t)


# Closed-loop
pid = PID(KP,KI,KD,phi_setpoint,time_step)
n_steps = len(t)
phi_feedback = X0[0]
X_old = X0
X_vector = [X0]
tx_vector = []
u_pid = u_(0)
f_t, t_x, t_y, t_z = u_pid

for i in range(1, n_steps):
    t_i = i*time_step
    t_i_old = (i-1)*time_step
    t_vector = np.linspace(t_i_old, t_i, 10)
    tx_pid = pid.compute(phi_feedback)
    #t_x = tx_pid
    X = odeint(f2, X_old,t_vector, args=(f_t, tx_pid, t_y, t_z))
    phi_feedback = X[-1][0]
    X_old = X[-1]
    X_vector.append(X[-1])
    tx_vector.append(tx_pid)



#print('pid',np.shape(X_vector))
#print(X_vector[5])
#plot_states(np.array(X_vector),t)
#quali_pid(np.array(X_vector),t)
#quali_torquex(np.array(X_vector),t)