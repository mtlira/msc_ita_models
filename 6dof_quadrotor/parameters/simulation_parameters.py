time_step = 2e-3 #5e-3 é um bom valor
T_sample = 2e-2 # MPC sample time
N = 60
M = 10

num_reference_outputs = 6 # x, y, z, phi, beta, psi

if time_step >= T_sample:
    raise ValueError('Time step is bigger than time sample')