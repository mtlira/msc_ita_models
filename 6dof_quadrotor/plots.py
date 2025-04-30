import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import os
import pandas as pd

class DataAnalyser(object):
    def __init__(self):
        self.dataset = None

    def plot_states(self, X,t, X_lin = None, trajectory = None, u_vector = None, omega_vector = None, equal_scales=False, legend = [], save_path = None):
        handles = []
        
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
        plt.subplots_adjust(left=0.083, bottom=0.083, right=0.948, top=0.914, wspace=0.23, hspace=0.31)
        
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
        plt.subplots_adjust(left=0.083, bottom=0.083, right=0.948, top=0.914, wspace=0.23, hspace=0.31)


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
        if save_path is None: plt.show()
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

        plt.subplots_adjust(left=0.125, bottom=0.071, right=0.921, top=0.96, wspace=0.195, hspace=0.279)


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

            plt.subplots_adjust(left=0.125, bottom=0.071, right=0.921, top=0.96, wspace=0.195, hspace=0.279)
            

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

                plt.subplots_adjust(left=0.125, bottom=0.071, right=0.921, top=0.96, wspace=0.195, hspace=0.279)

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
                
    def plot_rmse_histogram(self, mother_folder_path):
        pass#if self.dataset is None: self.dataset = self.load_datasets(mother_folder_path)

        
    def RMSe(self, position, trajectory):
        '''position and trajectory dimesnions: (N_iterations, 3)'''
        delta_position = trajectory - position   # shape (N, 3)
        squared_norms = np.sum(delta_position**2, axis=1)  # Sum over x, y, z for each time step
        RMSe = np.sqrt(np.mean(squared_norms))
        return RMSe


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
