import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pandas as pd
import seaborn as sns

class DataAnalyser(object):
    def __init__(self):
        self.dataset = None

    def plot_states(self, X,t, X_lin = None, trajectory = None, u_vector = None, omega_vector = None, equal_scales=False, legend = [], save_path = None, plot = True):
        handles = []
        #plt.tight_layout()
        # Rotation
        fig, axs = plt.subplots(2, 3)
        axs[0,0].plot(t,X[:,0])
        if X_lin is not None: axs[0,0].plot(t,X_lin[:,0])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[0,0].set_title('$\\phi(t)$')
        axs[0,0].set_xlabel('t (s)')
        axs[0,0].set_ylabel('$\\phi (rad)$')

        axs[0,1].plot(t,(-1)*X[:,1])
        if X_lin is not None: axs[0,1].plot(t,(-1)*X_lin[:,1])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[0,1].set_title('$\\theta(t)$')
        axs[0,1].set_xlabel('t (s)')
        axs[0,1].set_ylabel('$\\theta (rad)$')

        axs[0,2].plot(t,(-1)*X[:,2])
        if X_lin is not None: axs[0,2].plot(t,(-1)*X_lin[:,2])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[0,2].set_title('$\\psi(t)$')
        axs[0,2].set_xlabel('t (s)')
        axs[0,2].set_ylabel('$\\psi$ (rad)')

        axs[1,0].plot(t,X[:,3])
        if X_lin is not None: axs[1,0].plot(t,X_lin[:,3])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[1,0].set_title('p(t)')
        axs[1,0].set_xlabel('t (s)')
        axs[1,0].set_ylabel('p (rad/s)')

        axs[1,1].plot(t,(-1)*X[:,4])
        if X_lin is not None: axs[1,1].plot(t,(-1)*X_lin[:,4])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[1,1].set_title('q(t)')
        axs[1,1].set_xlabel('t (s)')
        axs[1,1].set_ylabel('q (rad/s)')

        axs[1,2].plot(t,(-1)*X[:,5])
        if X_lin is not None: axs[1,2].plot(t,(-1)*X_lin[:,5])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[1,2].set_title('r(t)')
        axs[1,2].set_xlabel('t (s)')
        axs[1,2].set_ylabel('r (rad/s)')
        fig.legend(legend if trajectory is None else legend[:-1]) # Because there is no reference in angle values
        #plt.subplots_adjust(left=0.083, bottom=0.083, right=0.948, top=0.914, wspace=0.23, hspace=0.31)
        fig.tight_layout()
        if save_path is not None: 
            plt.savefig(save_path + 'x_angular.png')
            for ax in axs.reshape(-1): ax.cla()
            plt.close(fig)
            del fig
            del axs

        # Translation
        fig, axs = plt.subplots(2, 3)
        axs[0,0].plot(t,X[:,6])
        if X_lin is not None: axs[0,0].plot(t,X_lin[:,6])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[0,0].set_title('u(t)')
        axs[0,0].set_xlabel('t (s)')
        axs[0,0].set_ylabel('u (m/s)')

        axs[0,1].plot(t,(-1)*X[:,7])
        if X_lin is not None: axs[0,1].plot(t,(-1)*X_lin[:,7])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[0,1].set_title('v(t)')
        axs[0,1].set_xlabel('t (s)')
        axs[0,1].set_ylabel('v (m/s)')

        axs[0,2].plot(t,(-1)*X[:,8])
        if X_lin is not None: axs[0,2].plot(t,(-1)*X_lin[:,8])
        if trajectory is not None: axs[0,2].plot(np.nan, np.nan, 'g--')
        axs[0,2].set_title('w(t)')
        axs[0,2].set_xlabel('t (s)')
        axs[0,2].set_ylabel('w (m/s)')

        handles.append(axs[1,0].plot(t,X[:,9])[0])
        if X_lin is not None: handles.append(axs[1,0].plot(t,X_lin[:,9])[0])
        if trajectory is not None: handles.append(axs[1,0].plot(t,trajectory[:,0], 'g--')[0])
        axs[1,0].set_title('x(t)')
        axs[1,0].set_xlabel('t (s)')
        axs[1,0].set_ylabel('x (m)')

        axs[1,1].plot(t,(-1)*X[:,10])
        if X_lin is not None: axs[1,1].plot(t,(-1)*X_lin[:,10])
        if trajectory is not None: axs[1,1].plot(t,-trajectory[:,1], 'g--')
        axs[1,1].set_title('y(t)')
        axs[1,1].set_xlabel('t (s)')
        axs[1,1].set_ylabel('y (m)')

        axs[1,2].plot(t,(-1)*X[:,11])
        if X_lin is not None: axs[1,2].plot(t,(-1)*X_lin[:,11])
        if trajectory is not None: axs[1,2].plot(t,-trajectory[:,2], 'g--')
        axs[1,2].set_title('z(t)')
        axs[1,2].set_xlabel('t (s)')
        axs[1,2].set_ylabel('z (m)')

        for i in range(len(handles)):
            handles[i] = mlines.Line2D([], [],
                                    color=handles[i].get_color(),
                                    linestyle=handles[i].get_linestyle(),
                                    label=f'c{i+1}' if i >= len(legend) else legend[i])
        
        fig.legend(handles=handles)
        fig.tight_layout()
        #plt.subplots_adjust(left=0.083, bottom=0.083, right=0.948, top=0.914, wspace=0.23, hspace=0.31)


        if save_path is not None: 
            plt.savefig(save_path + 'x_linear.png')
            for ax in axs.reshape(-1): ax.cla()
            plt.close(fig)
            del fig
            del axs

        fig = plt.figure()
        axs = plt.axes(projection='3d')
        axs.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
        if X_lin is not None: axs.plot3D(X_lin[:,9], X_lin[:,10]*(-1), X_lin[:,11]*(-1))
        if trajectory is not None: axs.plot3D(trajectory[:,0], -trajectory[:,1], -trajectory[:,2], 'g--')
        axs.set_xlabel('x (m)')
        axs.set_ylabel('y (m)')
        axs.set_zlabel('z (m)')
        axs.set_title('3D Plot')
        fig.legend(handles=handles)
        if equal_scales: axs.set_aspect('equal', adjustable='box')


        if save_path is not None: 
            plt.savefig(save_path + '3D.png')
            axs.cla()
            plt.close(fig)
            del fig
            del axs

        if trajectory is not None: legend = legend[:-1]
        self.plot_inputs(u_vector, t, legend, omega_vector, save_path)
        if plot: plt.show()
        plt.close('all')

    def plot_inputs(self, u_vector, t, legend, omega_vector = None, save_path=None):
        t = t[0:-1]
        handles = []
        if len(omega_vector[0]) <= 8: # It is a single input vector
            omega_list = [omega_vector]
            u_list = [u_vector]
        else: 
            u_list = u_vector
            omega_list = omega_vector

        fig, axs = plt.subplots(2, 2)
        for u in u_list: handles.append(axs[0,0].step(t,u[:,0])[0])
        axs[0,0].set_title('f_t (t)')
        axs[0,0].set_ylabel('f_t (N)')
        axs[0,0].set_xlabel('t (s)')

        for u in u_list: axs[0,1].step(t,u[:,1])
        axs[0,1].set_title('$\\tau_x (t)$')
        axs[0,1].set_ylabel('$\\tau_x (N.m)$')
        axs[0,1].set_xlabel('t (s)')

        for u in u_list: axs[1,0].step(t,u[:,2])
        axs[1,0].set_title('$\\tau_y (t)$')
        axs[1,0].set_ylabel('$\\tau_y (N.m)$')
        axs[1,0].set_xlabel('t (s)')

        for u in u_list: axs[1,1].step(t,u[:,3])
        axs[1,1].set_title('$\\tau_z (t)$')
        axs[1,1].set_ylabel('$\\tau_z (N.m)$')
        axs[1,1].set_xlabel('t (s)')

        for i in range(len(handles)):
            handles[i] = mlines.Line2D([], [],
                                    color=handles[i].get_color(),
                                    linestyle=handles[i].get_linestyle(),
                                    label=f'c{i+1}' if i >= len(legend) else legend[i])
        
        fig.legend(handles=handles)

        fig.tight_layout()
        #plt.subplots_adjust(left=0.125, bottom=0.071, right=0.921, top=0.96, wspace=0.195, hspace=0.279)


        if save_path is not None: 
            plt.savefig(save_path + 'inputs-forces.png')
            for ax in axs.reshape(-1): ax.cla()
            plt.close(fig)
            del fig
            del axs

        if omega_vector is not None:
            fig, axs = plt.subplots(2, 2)
            for omega in omega_list: axs[0,0].step(t,omega[:,0])
            axs[0,0].set_title('$\\omega_1 (t)$')
            axs[0,0].set_ylabel('$\\omega_1 (rad/s)$')
            axs[0,0].set_xlabel('t (s)')

            for omega in omega_list: axs[0,1].step(t,omega[:,1])
            axs[0,1].set_title('$\\omega_2 (t)$')
            axs[0,1].set_ylabel('$\\omega_2 (rad/s)$')
            axs[0,1].set_xlabel('t (s)')

            for omega in omega_list: axs[1,0].step(t,omega[:,2])
            axs[1,0].set_title('$\\omega_3 (t)$')
            axs[1,0].set_ylabel('$\\omega_3 (rad/s)$')
            axs[1,0].set_xlabel('t (s)')

            for omega in omega_list: axs[1,1].step(t,omega[:,3])
            axs[1,1].set_title('$\\omega_4 (t)$')
            axs[1,1].set_ylabel('$\\omega_4 (rad/s)$')
            axs[1,1].set_xlabel('t (s)')
        
            fig.legend(handles=handles)

            fig.tight_layout()
            #plt.subplots_adjust(left=0.125, bottom=0.071, right=0.921, top=0.96, wspace=0.195, hspace=0.279)
            

            if save_path is not None: 
                plt.savefig(save_path + 'inputs-rotors1.png')
                for ax in axs.reshape(-1): ax.cla()
                plt.close(fig)
                del fig
                del axs

            if np.shape(omega_list[0])[1] == 8:
                fig, axs = plt.subplots(2, 2)
                for omega in omega_list: axs[0,0].step(t,omega[:,4])
                axs[0,0].set_title('$\\omega_5 (t)$')
                axs[0,0].set_ylabel('$\\omega_5 (rad/s)$')
                axs[0,0].set_xlabel('t (s)')

                for omega in omega_list: axs[0,1].step(t,omega[:,5])
                axs[0,1].set_title('$\\omega_6 (t)$')
                axs[0,1].set_ylabel('$\\omega_6 (rad/s)$')
                axs[0,1].set_xlabel('t (s)')

                for omega in omega_list: axs[1,0].step(t,omega[:,6])
                axs[1,0].set_title('$\\omega_7 (t)$')
                axs[1,0].set_ylabel('$\\omega_7 (rad/s)$')
                axs[1,0].set_xlabel('t (s)')

                for omega in omega_list: axs[1,1].step(t,omega[:,7])
                axs[1,1].set_title('$\\omega_8 (t)$')
                axs[1,1].set_ylabel('$\\omega_8 (rad/s)$')
                axs[1,1].set_xlabel('t (s)')

                fig.legend(handles=handles)

                fig.tight_layout()
                #plt.subplots_adjust(left=0.125, bottom=0.071, right=0.921, top=0.96, wspace=0.195, hspace=0.279)

                if save_path is not None: 
                    plt.savefig(save_path + 'inputs-rotors2.png')
                    for ax in axs.reshape(-1): ax.cla()
                    plt.close(fig)
                    del fig
                    del axs

    def load_datasets(self, mother_folder_path):
        for subdir, _, files in os.walk(mother_folder_path):
            for file in files:
                if file == 'dataset_metadata.csv':
                    df = pd.read_csv(mother_folder_path + file, sep=',')
                    self.dataset = df if self.dataset is None else pd.concat([self.dataset, df])
        self.dataset.to_csv(mother_folder_path, sep='csv', index=False)
                
    def plot_rmse_histogram(self, dataset_path):

        df = pd.read_csv(dataset_path + 'dataset_metadata.csv', sep=',')
        min_value = np.min([np.min(df['mpc_RMSe']), np.min(df['nn_RMSe'])])
        max_value = np.max([np.max(df['mpc_RMSe']), np.max(df['nn_RMSe'])])
        bins = np.linspace(min_value, max_value, 30)

        plt.figure(figsize=(8,5))
        sns.histplot(df['mpc_RMSe'], bins=bins, color='royalblue', label='MPC', kde=True, stat='density', alpha=0.6)
        sns.histplot(df['nn_RMSe'], bins=bins, color='darkorange', label='Neural Network', kde=True, stat='density', alpha=0.6)
        plt.axvline(np.mean(df['mpc_RMSe']), color='royalblue', linestyle='--', linewidth=2)
        plt.axvline(np.mean(df['nn_RMSe']), color='darkorange', linestyle='--', linewidth=2)
        plt.xlabel('RMSE (m)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Comparison of RMSE Distributions of MPC and Neural Network', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_histogram(self, df, column_1, column_2, x_label, title, legend, normalization_column = None, colors=['royalblue','darkorange']):

        data_1 = df[column_1]
        data_2 = df[column_2]
        
        if normalization_column is not None:
            if isinstance(normalization_column, str):
                data_1 = data_1 / df[normalization_column]
                data_2 = data_2 / df[normalization_column]
            if isinstance(normalization_column, list) or isinstance(normalization_column, np.ndarray):
                for column in normalization_column:
                    data_1 = data_1 / df[column]
                    data_2 = data_2 / df[column]

        min_value = np.min([np.min(data_1), np.min(data_2)])
        max_value = np.max([np.max(data_1), np.max(data_2)])
        bins = np.linspace(min_value, max_value, 40)


        plt.figure(figsize=(8,5))
        sns.histplot(data_1, bins=bins, color=colors[0], label=legend[0], kde=True, stat='density', alpha=0.6)
        sns.histplot(data_2, bins=bins, color=colors[1], label=legend[1], kde=True, stat='density', alpha=0.6)
        plt.axvline(np.mean(data_1), color=colors[0], linestyle='--', linewidth=2)
        plt.axvline(np.mean(data_2), color=colors[1], linestyle='--', linewidth=1.5)
        if 'execution_time' in column_1 and 'num_iterations' in normalization_column:
            from parameters.simulation_parameters import T_sample
            plt.axvline(T_sample, color='black', linestyle='--', linewidth=2)
            x_min, x_max = plt.xlim()
            plt.text(T_sample + 0.01*(x_max - x_min), plt.ylim()[1]*0.9, 'Time sample', rotation=90,va='top', ha='left', color='black')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def stats_simulations(self, df, mpc_column, nn_column):        
        if mpc_column == 'mpc_execution_time_per_iteration' and nn_column == 'nn_execution_time_per_iteration':
            data_mpc = df['mpc_execution_time (s)'] / df['num_iterations']
            data_nn = df['nn_execution_time (s)'] / df['num_iterations']

        else:
            data_mpc = df[mpc_column]
            data_nn = df[nn_column]

        mean_mpc = data_mpc.mean()
        std_mpc = data_mpc.std()
        max_mpc = data_mpc.max()
        min_mpc = data_mpc.min()

        mean_nn = data_nn.mean()
        std_nn = data_nn.std()
        max_nn = data_nn.max()
        min_nn = data_nn.min()

        table = pd.DataFrame({
            'Controller': ['MPC', 'Neural Network'],
            'min': [min_mpc, min_nn],
            'max': [max_mpc, max_nn],
            'mean': [mean_mpc, mean_nn],
            'std': [std_mpc, std_nn],
        })
        return table


    def plot_histogram_temp(self, df, column_1, x_label, title, normalization_column = None, colors=['royalblue','darkorange']):
        data_1 = df[column_1]
        df = df.sort_values('inter_position_RMSe', axis=0, ascending=True)
        
        p75 = round(0.75*len(df))
        p90 = round(0.9*len(df))

        x75 = df.iloc[p75]['inter_position_RMSe']
        print('x75\n',x75)
        x90 = df.iloc[p90]['inter_position_RMSe']
        
        if normalization_column is not None:
            if isinstance(normalization_column, str):
                data_1 = data_1 / df[normalization_column]
            if isinstance(normalization_column, list) or isinstance(normalization_column, np.ndarray):
                for column in normalization_column:
                    data_1 = data_1 / df[column]

        min_value = np.min(np.min(data_1))
        max_value = np.max(np.max(data_1))
        bins = np.linspace(min_value, max_value, 30)


        plt.figure(figsize=(8,5))
        sns.histplot(data_1, bins=bins, color=colors[0], kde=True, stat='density', alpha=0.6)
        plt.axvline(np.mean(data_1), color=colors[0], linestyle='--', linewidth=2)
        x_min, x_max = plt.xlim()
        plt.axvline(x75, color='black', linestyle='--', linewidth=2)
        plt.text(x75 + 0.01*(x_max - x_min), plt.ylim()[1]*0.9, 'Percentile 75%', rotation=90,va='top', ha='left', color='black')
        plt.axvline(x90, color='black', linestyle='--', linewidth=2)
        plt.text(x90 + 0.01*(x_max - x_min), plt.ylim()[1]*0.9, 'Percentile 90%', rotation=90,va='top', ha='left', color='black')

        if 'execution_time' in column_1 and 'num_iterations' in normalization_column:
            from parameters.simulation_parameters import T_sample
            plt.axvline(T_sample, color='black', linestyle='--', linewidth=2)
            x_min, x_max = plt.xlim()
            plt.text(T_sample + 0.01*(x_max - x_min), plt.ylim()[1]*0.9, 'Time sample', rotation=90,va='top', ha='left', color='black')
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    


    def RMSe(self, position, trajectory):
        '''position and trajectory dimesnions: (N_iterations, 3)'''
        delta_position = trajectory - position   # shape (N, 3)
        squared_norms = np.sum(delta_position**2, axis=1)  # Sum over x, y, z for each time step
        RMSe = np.sqrt(np.mean(squared_norms))
        return RMSe
    
    def RMSe_control(self, u1, u2):
        return np.sqrt(np.mean((u2 - u1)**2, axis=0))

def plot_delays(X_nonlinear, trajectory, t, X_linear = False):
    samples_indexes = np.rint(np.linspace(0,1,11)*(len(t)-1)).astype('int')
    print('samples_indexes=',samples_indexes)
    delays = []
    samples_times = t[samples_indexes]
    for i in samples_indexes:
        j = i - 1
        min_error = 1e3
        error_j = min_error
        while min_error >= 0.1 and j < len(t) - 1:
            j += 1
            error_x_j = trajectory[i,0] - X_nonlinear[j,9]
            error_y_j = trajectory[i,1] - X_nonlinear[j,10]
            error_z_j = trajectory[i,2] - X_nonlinear[j,11]
            error_j = np.sqrt(error_x_j**2 + error_y_j**2 + error_z_j**2)
            if min_error > error_j:
                min_error = error_j
        if j < len(t) - 1:
            delays.append(t[j] - t[i])
        else: print ('Erro minimo acima do limite desejado ({error})'.format(error = min_error))
    fig = plt.figure()
    plt.plot(samples_times,delays)
    plt.xlabel('Time (s)')
    plt.ylabel('Delay (s)')
    plt.show()

def plot_errors(X_nonlinear, trajectory, t):
    fig, axs = plt.subplots(2, 2)
    axs[0,0].plot(t, trajectory[:,0] - X_nonlinear[:,9])
    axs[0,0].set_xlabel('Time (s)')
    axs[0,0].set_ylabel('Error in x (m)')

    axs[0,1].plot(t, trajectory[:,1] - X_nonlinear[:,10])
    axs[0,1].set_xlabel('Time (s)')
    axs[0,1].set_ylabel('Error in y (m)')

    axs[1,0].plot(t, trajectory[:,2] - X_nonlinear[:,11])
    axs[1,0].set_xlabel('Time(s)')
    axs[1,0].set_ylabel('Error in z (m)')
    plt.show()

################ QUALIFICATION PLOTS #################################################

def quali_shm(X,t):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t,(-1)*X[:,11])
    ax.set_title('z(t)')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (m)')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('3D Plot')
    plt.show()

def quali_pid(X,t):
    #PID control

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(t,X[:,0])
    ax.set_title('$\\phi(t)$')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('$\\phi (rad)$')

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('3D Plot')
    plt.show()

def quali_torquex(X,t):
    fig, axs = plt.subplots(ncols=2, nrows=2)
    gs = axs[1, 1].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1],projection='3d')

    axs[0,0].plot(t,(-1)*X[:,10])
    axs[0,0].set_title('y(t)')
    axs[0,0].set_xlabel('t (s)')
    axs[0,0].set_ylabel('y (m)')

    axs[1,0].plot(t,(-1)*X[:,11])
    axs[1,0].set_title('z(t)')
    axs[1,0].set_xlabel('t (s)')
    axs[1,0].set_ylabel('z (m)')

    axbig.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    axbig.set_xlabel('x (m)')
    axbig.set_ylabel('y (m)')
    axbig.set_zlabel('z (m)')
    axbig.set_title('3D Plot')
    plt.show()

def quali_linear(X,t,X_lin):
    fig, axs = plt.subplots(ncols=2, nrows=2)
    gs = axs[1, 1].get_gridspec()
    for ax in axs[0:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, -1],projection='3d')

    axs[0,0].plot(t,(-1)*X[:,10])
    axs[0,0].plot(t,(-1)*X_lin[:,10])
    axs[0,0].set_title('y(t)')
    axs[0,0].set_xlabel('t (s)')
    axs[0,0].set_ylabel('y (m)')

    axs[1,0].plot(t,(-1)*X[:,11])
    axs[1,0].plot(t,(-1)*X_lin[:,11])
    axs[1,0].set_title('z(t)')
    axs[1,0].set_xlabel('t (s)')
    axs[1,0].set_ylabel('z (m)')

    axbig.plot3D(X[:,9], X[:,10]*(-1), X[:,11]*(-1))
    axbig.plot3D(X_lin[:,9], X_lin[:,10]*(-1), X_lin[:,11]*(-1))
    axbig.set_xlabel('x (m)')
    axbig.set_ylabel('y (m)')
    axbig.set_zlabel('z (m)')
    axbig.set_title('3D Plot')

    fig.legend(['Non-linear','Linear'])
    plt.show()

if __name__ == '__main__':
    pass
    # c = DataAnalyser()
    # path = 'training_results\Training dataset v1 - octorotor/'
    # #c.plot_rmse_histogram('training_results\Training dataset v0 - octorotor/')
    # c.plot_histogram(path, 'mpc_RMSe', 'nn_RMSe', 'RMSE (m)','Comparison of RMSE Distributions of MPC and Neural Network', ['MPC', 'Neural Network'])
    # #c.plot_histogram('training_results\Training dataset v0 - octorotor/', 'mpc_execution_time (s)', 'nn_execution_time', '$t_{execution}/t_{simulation}$', 'Comparison of Execution Time Distributions', ['MPC', 'Neural Network'], normalization_column='simulation_time (s)')

    # c.plot_histogram(path, 'mpc_execution_time (s)', 'nn_execution_time (s)', '$t_{execution}/iteration$', 'Comparison of Execution Time Distributions', ['MPC', 'Neural Network'], normalization_column='num_iterations')
    # #c.plot_histogram('training_results\Training dataset v0 - octorotor/', 'mpc_execution_time (s)', 'nn_execution_time', 'CPU Use Percentage', 'Comparison of CPU Use Percentage', ['MPC', 'Neural Network'], normalization_column=['time_sample (s)', 'num_iterations'])

    # stats_rmse = c.stats_simulations(path, 'mpc_RMSe', 'nn_RMSe')
    # print(stats_rmse)

    # stats_execution_time = c.stats_simulations(path, 'mpc_execution_time_per_iteration', 'nn_execution_time_per_iteration')
    # print(stats_execution_time)