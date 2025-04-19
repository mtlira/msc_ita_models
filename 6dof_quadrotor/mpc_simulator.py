import numpy as np

class MPCSimulator(object):
    def __init__(self):
        self.X0 = X0
        self.time_step = time_step
        self.T_sample = T_sample

        pass

    def generate_simulations_batch(self):
        pass

    def simulate_mpc(self, time_step, T_sample, T_simulation, trajectory, restrictions, output_weights, control_weights, dataset_name, folder_name, disturb_input=False):
        '''
        Executes a control simulation with MPC given the time step, time sample, total simulation time and the desired trajectory.\n
        dataset_name: name of dataset folder the current simulation will belong to\n
        folder_name: name of the folder that will store the simulation results\n
        (obs: the folder hierarchy is dataset_name/folder_name/)
        '''
        t_samples = np.arange(0,T_simulation, T_sample)

        Bw = B @ model.Gama
        MPC = mpc.mpc(M, N, A, Bw, C, time_step, T_sample, output_weights, control_weights, restrictions)
        MPC.initialize_matrices()
        #try:
        x_mpc_rotors, u_rotors, omega_vector, NN_dataset, metadata = MPC.simulate_future_rotors(model, X0, t_samples, trajectory, omega_eq**2, generate_dataset=True, disturb_input=disturb_input)
        #except Exception as error:
        #    x_mpc_rotors, u_rotors, omega_vector, NN_dataset, metadata = None, None, None, None, None
        #    print('exception:', error)

        if x_mpc_rotors is not None:
            save_path = 'simulations/{}/{}/'.format(dataset_name, folder_name)
            #now = datetime.now()
            #current_time = now.strftime("%m_%d_%Hh-%Mm")
            Path(save_path).mkdir(parents=True, exist_ok=True)
            plot_states(x_mpc_rotors, t_samples[:np.shape(x_mpc_rotors)[0]], trajectory=trajectory, u_vector=u_rotors, omega_vector=omega_vector, equal_scales=True, legend=['Force/Moment optimization'], save_path=save_path)
            np.savetxt(save_path + "dataset.csv", NN_dataset, delimiter=",")
            return True, metadata
        return False, metadata
