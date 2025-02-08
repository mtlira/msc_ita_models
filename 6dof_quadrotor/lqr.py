import numpy as np
from scipy.integrate import odeint
from control import lqr, dlqr
from scipy.signal import StateSpace, cont2discrete, lsim, dlsim
from linearize import discretize
import euler

class LQR(object):
    def __init__(self, A, B, C, time_step, time_sample):
        self.A = np.array(A).astype(np.float64)
        self.B = np.array(B).astype(np.float64)
        self.C = np.array(C).astype(np.float64)
        Ad, Bd, _ = discretize(A, B, C, time_sample)
        self.Ad = Ad
        self.Bd = Bd
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

        self.integral_error = np.zeros(self.q) # Sum of (r-y)*dt
        self.output_error = np.zeros(self.q) # r - y

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

        # Discretization
        #Ad, Bd, _ = self.discretize()
        #self.A = Ad
        #self.B = Bd

        #self.A = np.eye(n_x) + self.time_sample*self.A
        #self.B = self.time_sample*self.B

        A_a = np.concatenate((self.A, np.zeros((n_x,q))), axis = 1)
        #A_a2 = np.concatenate((-self.C*self.time_sample, np.zeros((q,q))), axis = 1)
        A_a2 = np.concatenate((-self.C, np.zeros((q,q))), axis = 1)
        A_a = np.concatenate((A_a, A_a2), axis = 0)
        self.A_a = A_a

        B_a = np.concatenate((self.B, np.zeros((q,p))),axis = 0)
        self.B_a = B_a

        #G = np.concatenate((np.zeros((n_x,q)), self.time_sample*np.eye(q)), axis = 0)
        G = np.concatenate((np.zeros((n_x,q)), np.eye(q)), axis = 0)
        self.G = G

        # LQR augmented
        #x_max_aug = np.concatenate((x_max, np.ones(q)), axis = 0)
        x_max_aug = np.concatenate((x_max, x_max[-1-q+1:]), axis = 0)
        Q_aug = np.eye(n_x + q)
        for i in range(len(Q_aug)):
            Q_aug[i][i] = 1/(x_max_aug[i]**2)

        K_a, _, _ = lqr(A_a, B_a, Q_aug, R) # TODO: Talvez tenha que aumentar Q e R
        #K_a[:,n_x:] = K_a[:,n_x:]*1e-280
        self.K_a = K_a
        self.K = K_a[:,0:n_x] # LQR gain K from augmented LQR version
        self.K_i = K_a[:,n_x:] # Integrator gain
        print('K_a',K_a, 'dtype K_a', K_a.dtype)

        #self.C_a = np.array([ # TODO: AUTOMATIZAR CRIAÇÃO
        #                [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
        #                [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
        #                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]])
        
        self.C_a = np.concatenate((self.C, np.zeros((q, q))), axis = 1)


    def compute(self, reference, x_feedback):
        output = np.matmul(self.C,x_feedback)
        reference_error = reference - output
        #print('x_k', x_feedback, x_feedback.dtype,'integral_error',self.integral_error, self.integral_error.dtype,'reference_error',reference_error, reference_error.dtype,'output', output, output.dtype)
        #if np.shape(reference)[0] == 6: print(reference, reference_error)
        self.integral_error += reference_error*self.time_sample
        u = -np.matmul(self.K,x_feedback) - np.matmul(self.K_i,self.integral_error)
        return u
    
    # def simulate(self, X0, t, ref_vector, f, u_eq):
    #     self.integral_error = 0
    #     self.output_error = 0
    #     n_steps = len(t)
    #     x_feedback = X0
    #     X_vector = [X0]
    #     u_vector = []
    #     reference = ref_vector[0]

    #     for i in range(1, n_steps):
    #         u_i = u_eq + self.compute(reference, x_feedback)
    #         t_i = i*self.time_step
    #         t_i_old = (i - 1)*self.time_step
    #         t_vector = np.linspace(t_i_old, t_i, 10)
    #         f_t_i, t_x_i, t_y_i, t_z_i = u_i
    #         x_feedback = odeint(f, x_feedback, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
    #         #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
    #         x_feedback = x_feedback[-1]1
    #         reference = ref_vector[i]
    #         X_vector.append(x_feedback)
    #         u_vector.append(u_i)
            
    #     return np.array(X_vector), np.array(u_vector)

    def simulate2(self, X0, t_samples, ref_vector, f, u_eq):
        self.integral_error = 0
        self.output_error = 0
        x_k = X0
        X_vector = [X0]
        u_vector = []

        for k in range(0, len(t_samples) - 1):
            N = 0
            reference = ref_vector[np.min([k + N, len(ref_vector) - 1])]
            u_k = u_eq + self.compute(reference, x_k)
            f_t_i, t_x_i, t_y_i, t_z_i = u_k
            t_vector = np.arange(t_samples[k], t_samples[k+1], self.time_step) if self.time_step < self.time_sample else [t_samples[k], t_samples[k+1]]
            #t_vector = np.linspace(t_samples[k], t_samples[k+1], 100)
            x_k = odeint(f, x_k, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
            if np.linalg.norm(x_k[9:12]) > 100:
                print('LQR simulation stopped: x exploded')
                break
            #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_k = x_k[-1]
            X_vector.append(x_k)
            u_vector.append(u_k)

        return np.array(X_vector), np.array(u_vector)
    
    def simulate_speed_reference(self, X0, t_samples, ref_vector, f, u_eq):
        self.integral_error = 0
        self.output_error = 0
        x_k = X0
        X_vector = [X0]
        u_vector = []
        e = euler.Euler()

        for k in range(0, len(t_samples) - 1):
            N = 0
            reference = ref_vector[np.min([k + N, len(ref_vector) - 1])]

            # Rotation of speed vector to body reference axes
            v_G = reference[0:3] # Speed in global / inertial reference
            #vx_G = reference[-1]
            #vy_G = reference[-2]
            #vz_G = reference[-3]
            #v_G = np.array([vx_G, vy_G, vz_G])

            phi = x_k[0]
            theta = x_k[1]
            psi = x_k[2]

            v_I = e.R_global_to_body(phi, theta, psi) @ v_G
            # x y z são invertidos devido a definição de C
            reference[0] = 100*v_I[0]
            reference[1] = 100*v_I[1]
            reference[2] = 100*v_I[2]

            u_k = u_eq + self.compute(reference, x_k)
            f_t_i, t_x_i, t_y_i, t_z_i = u_k
            t_vector = np.arange(t_samples[k], t_samples[k+1], self.time_step) if self.time_step < self.time_sample else [t_samples[k], t_samples[k+1]]
            #t_vector = np.linspace(t_samples[k], t_samples[k+1], 100)
            x_k = odeint(f, x_k, t_vector, args = (f_t_i, t_x_i, t_y_i, t_z_i)) # TODO: Verificar se t_vector está correto (provavelmente não)
            if np.linalg.norm(x_k[9:12]) > 1000:
                print('LQR simulation stopped: x exploded')
                break
            #x_feedback = odeint(f, x_old, [t_i_old, t_i], args = (f_t_i, t_x_i, t_y_i, t_z_i))
            x_k = x_k[-1]
            X_vector.append(x_k)
            u_vector.append(u_k) 

        return np.array(X_vector), np.array(u_vector)   

    def simulate_linear(self, X0, t, r_tracking):
        self.integral_error = 0
        self.output_error = 0
        A_new = self.A_a - np.matmul(self.B_a, self.K_a)
        B_new = self.G
       
        #A_new = np.eye(len(A_new)) + self.time_sample*A_new
        #B_new = self.time_sample*B_new
        
        linear_sys_tracking = StateSpace(A_new, B_new, self.C_a, np.zeros((self.q,self.q)))
        X0_aug = np.concatenate((X0, np.zeros(self.q)), axis=0)

        _,_, x_tracking = lsim(linear_sys_tracking, r_tracking, t, X0 = X0_aug)
        return x_tracking
    
    def simulate_linear2(self, X0, t, r_tracking):
        self.integral_error = 0
        self.output_error = 0
        X0_aug = np.concatenate((X0, np.zeros(self.q)), axis=0)
        x_k = X0_aug
        A_eq = self.A_a - self.B_a @ self.K_a
        B_eq = self.G
        
        # Discretization
        A_eq, B_eq, _ = discretize(A_eq, B_eq, self.C_a, self.time_sample)
        #A_eq = np.eye(len(A_eq)) + self.time_sample*A_eq
        #B_eq = self.time_sample*B_eq

        x_vector = [X0]
        for k in range(0, len(t) - 1):
            #print('A_eq @ x_k', A_eq @ x_k, 'B_eq @ r_tracking[k]', B_eq @ r_tracking[k])
            x_k = A_eq @ x_k + B_eq @ r_tracking[k]
            x_vector.append(x_k[0:self.n_x]) # Tirando a parte aumentada
        return np.array(x_vector)
    
    def simulate_linear3(self, X0, t, r_tracking):
        self.integral_error = np.zeros(self.q) # Sum of (r-y)*dt
        self.output_error = np.zeros(self.q) # r - y
        x_k = X0
        x_vector = [X0]
        u_vector = []

        for k in range(0, len(t) - 1):
            u_k = self.compute(r_tracking[k], x_k)
            x_k = self.Ad @ x_k + self.Bd @ u_k
            if np.linalg.norm(x_k[9:12]) > 1000:
                print('LQR simulation stopped: x exploded')
                break
            x_vector.append(x_k) # Tirando a parte aumentada
            u_vector.append(u_k)
        return np.array(x_vector), np.array(u_vector)
    
    def simulate_linear4_speed_reference(self, X0, t, r_tracking):
        self.integral_error = np.zeros(self.q) # Sum of (r-y)*dt
        self.output_error = np.zeros(self.q) # r - y
        x_k = X0
        x_vector = [X0]
        u_vector = []
        e = euler.Euler()

        for k in range(0, len(t) - 1):
            reference = r_tracking[k]

            # Rotation of speed vector to body reference axes
            v_G = reference[0:3] # Speed in global / inertial reference

            phi = x_k[0]
            theta = x_k[1]
            psi = x_k[2]

            v_I = e.R_global_to_body(phi, theta, psi) @ v_G
            # x y z são invertidos devido a definição de C
            reference[0] = 100*v_I[0]
            reference[1] = 100*v_I[1]
            reference[2] = 100*v_I[2]

            u_k = self.compute(reference, x_k)
            x_k = self.Ad @ x_k + self.Bd @ u_k
            if np.linalg.norm(x_k[9:12]) > 1000:
                print('LQR simulation stopped: x exploded')
                break
            x_vector.append(x_k) # Tirando a parte aumentada
            u_vector.append(u_k)
        return np.array(x_vector), np.array(u_vector)
    def discretize(self):
        #sys = StateSpace(self.A,self.B,self.C, np.zeros((q,p)))
        sys_d = cont2discrete((self.A,self.B,self.C, np.zeros((self.q,self.p))), self.time_sample, 'zoh')
        Ad, Bd, Cd, _, _ = sys_d
        return Ad, Bd, Cd
    

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