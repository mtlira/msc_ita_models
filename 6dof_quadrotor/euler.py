import numpy as np
class Euler(object):
    def _init_(self):
        pass
    def R_body_to_global(self, phi, theta, psi):
        '''
        Extrinsic rotation matrix.
        phi: roll angle (rad)
        theta: pitch angle (rad)
        psi: yaw angle (rad)
        '''

        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R = np.array([
            [c_theta*c_psi, s_phi*s_theta*c_psi - c_phi*s_psi, c_phi*s_theta*c_psi + s_phi*s_psi],
            [c_theta*s_psi, s_phi*s_theta*s_psi + c_phi*c_psi, c_phi*s_theta*s_psi - s_phi*c_psi],
            [-s_theta, s_phi*c_theta, c_phi*c_theta]
        ])
        return R
    
    def R_global_to_body(self, phi, theta, psi):
        return np.linalg.inv(self.R_body_to_global(phi, theta, psi))