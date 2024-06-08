from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

m = 1
g = 9.80665
J = 0.2

t = np.arange(0,20, 1e-3)

def u(t):
    return [m*g + np.sin(2*np.pi*0.1*t), 0]

def f(x,t):
    dx_dt = [0, 0, 0, 0, 0, 0]
    dx_dt[0] = x[1]
    dx_dt[1] = 1/m*u(t)[0]*np.sin(x[4])
    dx_dt[2] = x[3]
    dx_dt[3] = 1/m*u(t)[0]*np.cos(x[4]) - g
    dx_dt[4] = x[5]
    dx_dt[5] = 1/J*u(t)[1]
    return dx_dt

X = odeint(f, y0=[0,0,0,0,0,0], t=t)

fig, axs = plt.subplots(2, 3)
axs[0,0].plot(t,X[:,0])
axs[0,0].set_title('x(t)')

axs[0,1].plot(t,X[:,1])
axs[0,1].set_title('$\dot x(t)$')

axs[0,2].plot(t,X[:,2])
axs[0,2].set_title('z(t)')

axs[1,0].plot(t,X[:,3])
axs[1,0].set_title('$\dot z(t)$')

axs[1,1].plot(t,X[:,4]*180/np.pi)
axs[1,1].set_title('$\\theta$')

axs[1,2].plot(t,X[:,5]*180/np.pi)
axs[1,2].set_title('$\dot \\theta(t)$')

plt.figure()
plt.plot(X[:,0], X[:,2])
plt.show()
