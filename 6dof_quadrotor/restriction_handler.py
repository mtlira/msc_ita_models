import numpy as np

class Restriction(object):
    def __init__(self, model, time_sample, N, M):
        self.model = model
        self.time_sample = time_sample
        self.N = N
        self.M = M


    def restriction(self, operation_mode, rotors_idx=None):
        if operation_mode not in ['normal', 'total_failure', 'partial_failure']:
            raise ValueError('Operation mode not correct')
        
        omega_eq = self.model.get_omega_eq()

        omega_max = np.sqrt(2)*omega_eq # Decision: Max thrust force = 2*m*g => omega_max = sqrt(2)*omega_eq
        omega_min = np.zeros(self.model.num_rotors)

        delta_omega_max = self.model.angular_acceleration*np.ones(self.model.num_rotors) * self.time_sample
        delta_omega_min = (-1) * delta_omega_max

        if operation_mode != 'normal':
            # Failure constraint
            for idx in rotors_idx:
                if operation_mode == 'total_failure':
                    omega_max[idx] = 0
                else: # partial failure
                    omega_max[idx] = (0.8*self.random.rand() + 0.1)*omega_eq[idx]

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
    
    def generate_restrictions(self):

        restrictions = []

        # Normal operation
        restrictions.append(self.restriction('normal'))

        # Single total failures
        for i in range(self.model.num_rotors):
            restrictions.append(self.restriction('total_failure', [i]))

        # 2 total failures of all rotors
        for i in range(self.model.num_rotors):
            restrictions.append(self.restriction('total_failure', [i]))