import numpy as np
from scipy.integrate import odeint
from control import lqr
from scipy.signal import StateSpace, lsim

class LQR(object):
    def __init__(self, A, B, C, time_step, time_sample):
        self.A = A
        self.B = B
        self.C = C
        #self.K = K
        #self.K_i = K_i # Integral contgrol gain
        self.integral_error = 0 # Sum of (r-y)*dt
        self.output_error = 0 # r - y
        self.time_step = time_step # Simulation time
        self.time_sample = time_sample # Controller time sample

        # p - Number of controls
        # q - Number of outputs
        # n_x - Number of states
        self.p = np.shape(self.B)[1]
        print('p =',self.p,'Controls')
        self.q = np.shape(self.C)[0]
        print('q =',self.q,'Outputs')
        self.n_x = np.shape(self.A)[0]

    def initialize(self, x_max, u_max):
        """
        Builds weight matrices Q and R using Bryson's rule the LQR gain K and the augmented matrices and gain (K_aug) needed for integral action.
        """
        p = self.p
        q = self.q
        n_x = self.n_x

        Q = np.eye(n_x)
        for i in range(len(Q)):
            Q[i][i] = 1/(x_max[i]**2)
        self.Q = Q

        R = np.eye(p)
        for i in range(len(R)):
            R[i][i] = 1/(u_max[i]**2)
        self.R = R

        A_a = np.concatenate((self.A, np.zeros((n_x,q))), axis = 1)
        A_a2 = np.concatenate((-self.C, np.zeros((q,q))), axis = 1)
        A_a = np.concatenate((A_a, A_a2), axis = 0)
        self.A_a = A_a

        B_a = np.concatenate((self.B, np.zeros((q,p))),axis = 0)
        self.B_a = B_a

        G = np.concatenate((np.zeros((n_x,q)), np.eye(q)), axis = 0)
        self.G = G

        # LQR augmented
        x_max_aug = np.concatenate((x_max, np.ones((q))), axis = 0)
        Q_aug = np.eye(n_x + q)
        for i in range(len(Q_aug)):
            Q_aug[i][i] = 1/(x_max_aug[i]**2)

        K_a, _, _ = lqr(A_a, B_a, Q_aug, R) # TODO: Talvez tenha que aumentar Q e R
        self.K_a = K_a
        self.K = K_a[:,0:n_x] # LQR gain K from augmented LQR version
        self.K_i = K_a[:,n_x:] # Integrator gain

        #self.C_a = np.array([ # TODO: AUTOMATIZAR CRIAÇÃO
        #                [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        #                [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
        #                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])
        
        self.C_a = np.concatenate((self.C, np.zeros((q, q))), axis = 1)


    def compute(self, reference, x_feedback):
        output = np.matmul(self.C,x_feedback)
        reference_error = reference - output
        self.integral_error += reference_error*self.time_sample
        u = -np.matmul(self.K,x_feedback) - np.matmul(self.K_i,self.integral_error)
        return u
    
    def simulate(self, X0, t, ref_vector, f, u_eq):
        self.integral_error = 0
        self.output_error = 0
        n_steps = len(t)
        x_feedback = X0
        X_vector = [X0]
        u_vector = []
        reference = ref_vector[0]

        for i in range(1, n_steps):
            u_i = u_eq + self.compute(reference, x_feedback)
            t_i = i*self.time_step
            t_i_old = (i - 1)*self.time_step
            t_vector = np.linspace(t_i_old, t_i, 10)
            f_t_i, t_x_i, t_y_i, t_z_i = u_i
            x_feedback = odeint(f, x_feedback, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
            #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_feedback = x_feedback[-1]
            reference = ref_vector[i]
            X_vector.append(x_feedback)
            u_vector.append(u_i)
            
        return np.array(X_vector), np.array(u_vector)

    def simulate2(self, X0, t_samples, ref_vector, f, u_eq):
        self.integral_error = 0
        self.output_error = 0
        x_k = X0
        X_vector = [X0]
        u_vector = []

        for k in range(0, len(t_samples) - 1):
            reference = ref_vector[k]
            u_k = u_eq + self.compute(reference, x_k)
            f_t_i, t_x_i, t_y_i, t_z_i = u_k
            t_vector = np.arange(t_samples[k], t_samples[k+1], self.time_step) if self.time_step < self.time_sample else [t_samples[k], t_samples[k+1]]
            #t_vector = np.linspace(t_samples[k], t_samples[k+1], 100)
            x_k = odeint(f, x_k, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
            #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_k = x_k[-1]
            X_vector.append(x_k)
            u_vector.append(u_k)

        return np.array(X_vector), np.array(u_vector)

    def simulate_linear(self, X0, t, r_tracking):
        self.integral_error = 0
        self.output_error = 0
        linear_sys_tracking = StateSpace(self.A_a - np.matmul(self.B_a, self.K_a), self.G, self.C_a, np.zeros((self.q,self.q)))
        X0_aug = np.concatenate((X0, np.zeros(self.q)), axis=0)

        _,_, x_tracking = lsim(linear_sys_tracking, r_tracking, t, X0 = X0_aug)
        return x_tracking
    

#DRAFT
# LQR - state regulation only (no tracking)
#linear_sys = signal.StateSpace(A - B*K,B,np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
#_, _, delta_xout = signal.lsim(linear_sys, 0, t, X0 = X0 - X_eq)
#xout = X_eq + delta_xout

#eig_Acl, _ = np.linalg.eig(np.array(A-B*K, dtype='float'))
#print('Eigenvalues of A - B*K (closed-loop):')
#print(eig_Acl)
#plot_states(xout, t)

#linear_sys2 = ct.StateSpace(A - B*K, B, np.eye(np.shape(A)[0]), np.zeros((np.shape(A)[0],np.shape(B)[1])))
#ct_out = ct.forced_response(linear_sys2, t, 0, X0 - X_eq, transpose=True)
#xout2 = X_eq + ct_out.states

#eig_Acl_a, _ = np.linalg.eig(np.array(A_a - np.matmul(B_a,K_a), dtype='float'))
#print('Eigenvalues of Aa - Ba*Ka (closed-loop):')
#print(eig_Acl_a)