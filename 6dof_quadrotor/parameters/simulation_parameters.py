time_step = 1e-3 #5e-3 é um bom valor
T_sample = 1e-2 # MP sample time
N = 100
M = 10

num_outputs = 6 # x, y, z, phi, beta, psi

if time_step >= T_sample:
    raise ValueError('Time step is bigger than time sample')