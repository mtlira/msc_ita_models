import numpy as np
from scipy.integrate import odeint

class LQR(object):
    def __init__(self, K, K_i, C, time_step):
        self.K = K
        self.K_i = K_i # Integral contgrol gain
        self.integral_error = 0 # Sum of (r-y)*dt
        self.output_error = 0 # r - y
        self.C = C
        self.time_step = time_step

    def compute(self, reference, x_feedback):
        output = np.matmul(self.C,x_feedback)
        reference_error = reference - output
        self.integral_error += reference_error*self.time_step
        u = -np.matmul(self.K,x_feedback) - np.matmul(self.K_i,self.integral_error)
        return u
    
    def simulate(self, X0, t, ref_vector, f, u_eq):
        n_steps = len(t)
        x_feedback = X0
        x_old = X0
        X_vector = [X0]
        u_vector = []
        reference = ref_vector[0]

        for i in range(1, n_steps):
            t_i = i*self.time_step
            t_i_old = (i - 1)*self.time_step
            t_vector = np.linspace(t_i_old, t_i, 10)
            u_i = u_eq + self.compute(reference, x_feedback)
            f_t_i, t_x_i, t_y_i, t_z_i = u_i
            x_feedback = odeint(f, x_old, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_feedback = x_feedback[-1]
            reference = ref_vector[i]
            x_old = x_feedback #maybe remove x_old. Not necessary!
            X_vector.append(x_feedback)
            u_vector.append(u_i)
            
        return np.array(X_vector), np.array(u_vector)