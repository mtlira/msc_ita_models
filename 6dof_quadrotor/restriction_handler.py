import numpy as np

class Restriction(object):
    def __init__(self, model, time_sample, N, M):
        self.model = model
        self.time_sample = time_sample
        self.N = N
        self.M = M


    def normal_restriction(self):
        omega_eq = self.model.get_omega_eq()

        omega_max = np.sqrt(2)*omega_eq # Decision: Max thrust force = 2*m*g => omega_max = sqrt(2)*omega_eq
        omega_min = np.zeros(self.model.num_rotors)
        
        delta_omega_max = self.model.angular_acceleration*np.ones(self.model.num_rotors) * self.time_sample
        delta_omega_min = (-1) * delta_omega_max

        # Order: x, y, z, phi, beta, psi
        y_max = np.array([50, 50, 50, 1.4, 1.4, 1.4]) # Tune in case of unsatisfactory simulation results
        y_min = (-1) * y_max

        restrictions = {
            'delta_omega_max': delta_omega_max,
            'delta_omega_min': delta_omega_min,
            'omega_max': omega_max,
            'omega_min': omega_min,
            'y_max': y_max,
            'y_min': y_min
        }

        # Order: x, y, z, phi, beta, psi
        delta_y_max = np.array([1, 1, 1, 0.8, 0.8, 0.8]) # Tune in case of unsatisfactory simulation results


        output_weights = 1 / (self.N*delta_y_max**2) # Deve variar a cada passo de simulação?
        control_weights = 1 / (self.M*restrictions['delta_omega_max']**2)

        return restrictions, output_weights, control_weights