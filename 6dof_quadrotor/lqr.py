import numpy as np
from scipy.integrate import odeint

class LQR(object):
    def __init__(self, K, K_i, C, time_step, time_sample):
        self.K = K
        self.K_i = K_i # Integral contgrol gain
        self.integral_error = 0 # Sum of (r-y)*dt
        self.output_error = 0 # r - y
        self.C = C
        self.time_step = time_step # Simulation time
        self.time_sample = time_sample # Controller time sample

    def compute(self, reference, x_feedback):
        output = np.matmul(self.C,x_feedback)
        reference_error = reference - output
        self.integral_error += reference_error*self.time_step
        u = -np.matmul(self.K,x_feedback) - np.matmul(self.K_i,self.integral_error)
        return u
    
    def simulate(self, X0, t, ref_vector, f, u_eq):
        n_steps = len(t)
        x_feedback = X0
        X_vector = [X0]
        u_vector = []
        reference = ref_vector[0]
        print(self.time_step,2*self.time_step,3*self.time_step)

        for i in range(1, n_steps):
            u_i = u_eq + self.compute(reference, x_feedback)
            t_i = i*self.time_step
            t_i_old = (i - 1)*self.time_step
            t_vector = np.linspace(t_i_old, t_i, 10)
            f_t_i, t_x_i, t_y_i, t_z_i = u_i
            x_feedback = odeint(f, x_feedback, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
            #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_feedback = x_feedback[-1]
            reference = ref_vector[i]
            X_vector.append(x_feedback)
            u_vector.append(u_i)
            
        return np.array(X_vector), np.array(u_vector)

    def simulate2(self, X0, t_samples, ref_vector, f, u_eq):
        x_k = X0
        X_vector = [X0]
        u_vector = []
        print(t_samples[1],t_samples[2],t_samples[3])

        for k in range(0, len(t_samples) - 1):
            reference = ref_vector[k]
            u_k = u_eq + self.compute(reference, x_k)
            f_t_i, t_x_i, t_y_i, t_z_i = u_k
            t_vector = np.arange(t_samples[k], t_samples[k+1], self.time_step) if self.time_sample < self.time_step else [t_samples[k], t_samples[k+1]]
            #t_vector = np.linspace(t_samples[k], t_samples[k+1], 10)
            x_k = odeint(f, x_k, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
            #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_k = x_k[-1]
            X_vector.append(x_k)
            u_vector.append(u_k)

        return np.array(X_vector), np.array(u_vector)