import numpy as np

class Restriction(object):
    def __init__(self, num_rotors):
        self.num_rotors = num_rotors

    def normal_restriction(self, omega_eq):

        omega_max = np.sqrt(2)*omega_eq # Decision: Max thrust force = 2*m*g => omega_max = sqrt(2)*omega_eq
        u_max = omega_max**2 - omega_eq**2

        omega_min = np.zeros(self.num_rotors)
        u_min = omega_min**2 - omega_eq**2

        # restrictions = {
        #     "delta_u_max": np.linalg.pinv(model.Gama) @ [10*m*g*T_sample, 0, 0, 0],
        #     "delta_u_min": np.linalg.pinv(model.Gama) @ [-10*m*g*T_sample, 0, 0, 0],
        #     "u_max": u_max,
        #     "u_min": u_min,
        #     "y_max": np.array([20, 20, 20, 1.4, 1.4, 1.4]),
        #     "y_min": np.array([-20, -20, -20, -1.4, -1.4, -1.4]),
        # }