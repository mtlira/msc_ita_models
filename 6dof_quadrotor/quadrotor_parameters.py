m = 10
g = 9.80665
I_x = 0.8
I_y = 0.8
I_z = 0.8

### Control allocation parameters ###
l = 1 # multirotor's arm (distance from the center to the propeller)
b = m*g/(200**2) # f_i = b*w^2, f_i is force of propeller and w is angular speed
#k_t = 0.01 # Torque = k_t * Tração, entre 0.01 e 0.03 (segundo internet)
k_t = 0.0001 # Valor de k_t afeta na velocidade de divergência de psi(t)
d = b*k_t # Torque = d * w^2