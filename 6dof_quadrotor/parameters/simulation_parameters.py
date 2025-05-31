time_step = 2e-3 #5e-3 é um bom valor
T_sample = 2e-2 # MPC sample time
N = 60
M = 10

include_phi_theta_reference = False
include_psi_reference = True
gain_scheduling = False

num_reference_outputs = 4 # x, y, z, psi

if time_step >= T_sample:
    raise ValueError('Time step is bigger than time sample')