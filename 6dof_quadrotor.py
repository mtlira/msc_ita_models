from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

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

def u_(t):
    return [1.1*m*g, 0, 0, 0]

def f(X,t):
    phi, theta, psi, p, q, r, u, v, w, x, y, z = X
    f_t, t_x, t_y, t_z = u_(t)
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

X = odeint(f, y0=[0,0,0,0,0,0,0,0,0,0,0,0], t=t)

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