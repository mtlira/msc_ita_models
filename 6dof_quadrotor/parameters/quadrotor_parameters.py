from numpy import pi

use_old_parameters = False
if use_old_parameters:
#Old parameters
    m = 10
    g = 9.80665
    I_x = 0.8
    I_y = 0.8
    I_z = 0.8
    l = 1
    b = 1e-4 # f_i = b*w^2, f_i is force of propeller and w is angular speed
    k_t = 0.0001 # Valor de k_t afeta na velocidade de divergência de psi(t)
    d = b*k_t # Torque = d * w^2
    num_rotors = 4
    thrust_to_weight = 2.5

else:
# Parameters from reference
    m = 0.65
    g = 9.80665
    I_x = 7.5e-3
    I_y = 7.5e-3
    I_z = 1.3e-2
    rho = 1.293
    CT = 0.055
    CQ = 0.024
    R = 0.15
    A = pi*R**2
    l = 0.232
    #b = 1.5e-4 #1e-4 # f_i = bw^2, f_i is force of propeller and w is angular speed
    #d = 7.4e-6 #7.4e-6 # t_i = d*w^2 t_i is torque of propeller 
    b = rho*CT*A*R**2
    d = rho*CQ*A*R**3
    num_rotors = 4
    thrust_to_weight = 2.5

### Control allocation parameters ###
#l = 1 # multirotor's arm (distance from the center to the propeller)
#b = 1e-4 # f_i = b*w^2, f_i is force of propeller and w is angular speed
#k_t = 0.01 # Torque = k_t * Tração, entre 0.01 e 0.03 (segundo internet)
#k_t = 0.0001 # Valor de k_t afeta na velocidade de divergência de psi(t)
#d = b*k_t # Torque = d * w^2
#d = 7.4e-6