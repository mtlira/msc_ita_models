import numpy as np
from scipy.integrate import odeint

class PID(object):
    def __init__(self, KP, KI, KD, setpoint, time_step):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.setpoint = setpoint
        self.error = 0
        self.proportional_error = 0
        self.previous_error = 0
        self.integral_error = 0
        self.derivative_error = 0
        self.output = 0
        self.time_step = time_step

    def compute(self, feedback):
        self.error = self.setpoint - feedback
        self.proportional_error = self.KP*self.error
        self.integral_error += self.KI*self.error*self.time_step
        self.derivative_error = self.KD*(self.error - self.previous_error)/self.time_step
        self.previous_error = self.error
        self.output = self.proportional_error + self.integral_error + self.derivative_error
        return self.output
    
    def simulate(self, t, X0, u_, time_step, f):
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
            tx_pid = self.compute(phi_feedback)
            #t_x = tx_pid
            X = odeint(f, X_old,t_vector, args=(f_t, tx_pid, t_y, t_z)) # Only tx is being controlled
            phi_feedback = X[-1][0]
            X_old = X[-1]
            X_vector.append(X[-1])
            tx_vector.append(tx_pid)
        return X_vector

