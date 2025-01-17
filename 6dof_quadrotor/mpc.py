import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.signal import StateSpace, cont2discrete, lsim
import cvxopt

class mpc(object):
    def __init__(self, M, N, rho, A, B, C, time_step, T_sample, output_weights, control_weights, restrictions):
        self.M = M
        self.N = N
        self.rho = rho
        self.A = A
        self.B = B
        self.C = C
        self.T = time_step
        self.T_sample = T_sample
        self.output_weights = output_weights
        self.control_weights = control_weights
        self.restrictions = restrictions

    def initialize_matrices(self):
        # 1. Discretization of the space state

        # Discretização aproximada
        #Ad = np.eye(np.shape(self.A)[0]) + self.A*self.T
        #Bd = self.B*self.T

        # Discretização exata
        Ad, Bd, _ = self.discretize()


        # #################################### APAGAR #################################################
        #Ad = self.A
        #Bd = self.B
        # #################################### APAGAR #################################################

        # 2. Initialize A_tilda, B_tilda, C_tild and phi
        A_tilda = np.concatenate((
            np.concatenate((Ad, Bd), axis = 1), # TODO coonfirm if its really Ad and Bd or A and B
            np.concatenate((np.zeros((4,12)), np.eye(4)), axis = 1)
            ), axis = 0)

        B_tilda = np.concatenate((Bd, np.eye(4)), axis = 0)

        C_tilda = np.concatenate((self.C, np.zeros((3,4))), axis = 1)

        phi = C_tilda @ A_tilda
        for i in range(2, self.N+1):
            phi = np.concatenate((phi, C_tilda @ np.linalg.matrix_power(A_tilda, i)), axis = 0)
        print('shape phi',np.shape(phi))

        # 3. Initialize G
        # First column
        column = C_tilda @ B_tilda
        for i in range(1,self.N):
            column = np.concatenate((column, C_tilda @ np.linalg.matrix_power(A_tilda, i) @ B_tilda), axis = 0)

        # Other columns
        G = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, 3, axis = 0)
            column[0:3,0:4] = np.zeros((3,4))
            G = np.concatenate((G, column), axis = 1)

        # 4. Initialize Q and R
        Q = np.diag(np.full(3, self.output_weights))
        print('Q=',Q)
        R = np.diag(np.full(4, self.control_weights))
        print('R=',R)

        # Q - First column
        column = np.array(Q, copy = True)
        for i in range(1, self.N):
            column = np.concatenate((column, np.zeros((3,3))), axis = 0)
        
        # Q - Other columns
        Q_super = np.array(column, copy = True)
        for i in range(1, self.N):
            column = np.roll(column, 3, axis = 0)
            column[0:3, 0:3] = np.zeros((3,3))
            Q_super = np.concatenate((Q_super, column), axis = 1)
        #print('Q_super=\n',pd.DataFrame(Q_super))
        #print('diag Q_super =',np.diag(Q_super))
        #print('shape Qsuper =',np.shape(Q_super))

        # R - First column
        column = np.array(R, copy = True)
        for i in range(1, self.M):
            column = np.concatenate((column, np.zeros((4,4))), axis = 0)
        
        # R - Other columns
        R_super = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, 4, axis = 0)
            column[0:4, 0:4] = np.zeros((4,4))
            R_super = np.concatenate((R_super, column), axis = 1)
        #print('R_super=\n',pd.DataFrame(R_super))
        #print('diag R_super =',np.diag(R_super))
        #print('shape Rsuper =',np.shape(R_super))

        # Gn, Hqp and Aqp
        Gn = Q_super @ G
        Hqp = 2*(np.transpose(G) @ Q_super @ G + R_super)

        # Aqp
            # T_M
        column = np.eye(4)
        for i in range(1,self.M):
            column = np.concatenate((column, np.eye(4)), axis = 0)
        
        T_M = np.array(column, copy = True)
        for i in range(1, self.M):
            column = np.roll(column, 4, axis = 0)
            column[0:4, 0:4] = np.zeros((4,4))
            T_M = np.concatenate((T_M, column), axis = 1)

        print(np.diag(T_M))
        print('T_M =\n',pd.DataFrame(T_M))
        print('shape T_M =',np.shape(T_M))

        Aqp = np.eye(4*self.M)
        Aqp = np.concatenate((
            Aqp,
            -np.eye(4*self.M),
            T_M,
            -T_M,
            G,
            -G
            ), axis = 0)
        
        print('Aqp =\n', pd.DataFrame(Aqp))
        print('shape Aqp =',np.shape(Aqp))

        self.Ad = Ad
        self.Bd = Bd
        self.A_tilda = A_tilda
        self.B_tilda = B_tilda
        self.C_tilda = C_tilda,
        self.phi = phi
        self.G = G
        self.Gn = Gn
        self.Q_super = Q_super
        self.R_super = R_super
        self.Aqp = Aqp
        self.Hqp = Hqp

    #def cost_function(x, Hqp, fqp):
    #    return 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x
    
    #def constraints(x, Aqp, bqp):
    #    return -(Aqp @ x - bqp)
    
    def simulate(self, f_model, X0, t_samples, trajectory, u_eq):
        #u_minus_1 = np.array(u_eq) # TODO: Confirmar se é u_eq ou 0
        u_minus_1 = np.zeros(4)
        x_k = X0
        u_k_minus_1 = u_minus_1
        X_vector = [X0]
        u_vector = []
        Hqp = self.Hqp
        cvxopt.solvers.options['show_progress'] = False

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            #ref_N = trajectory[k:k+self.N] # TODO validar se termina em k+N-1 ou em k+N
            ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = self.phi @ epsilon_k
            fqp = 2*np.transpose(self.Gn) @ (f - ref_N)
            
            # bqp
            bqp = np.concatenate((
                np.tile(self.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(self.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(self.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - self.restrictions['u_min'], self.M),
                np.tile(self.restrictions['y_max'], self.N) - f,
                f - np.tile(self.restrictions['y_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            self.Aqp = self.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(self.Aqp), cvxopt.matrix(bqp))
            x = np.array(res['x']).reshape((Hqp.shape[1],))
            ##################################################


            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(4), np.zeros((4, 4*(self.M - 1)))), axis = 1) @ x # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq

            # Apply control u_k in the multi-rotor
            f_t_k, t_x_k, t_y_k, t_z_k = u_k + u_eq # Attention for u_eq (solved the problem)
            t_simulation = np.arange(t_samples[k], t_samples[k+1], self.T)
            #t_simulation2 = np.arange(0,t_samples[1], self.T)
            #x_k = odeint(f_model, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = odeint(f_model, x_k, t_simulation, args = (f_t_k, t_x_k, t_y_k, t_z_k))
            x_k = x_k[-1]
            #if np.linalg.norm(x_k[9:12]) > 10 or np.max(np.abs(x_k[0:2])) > 1.4:
            #    print('Simulation exploded.')
            #    print('x_k =',x_k)
            #    break
            
            X_vector.append(x_k)
            u_k_minus_1 = u_k
            u_vector.append(u_k)
            #delta_u_initial = np.tile(delta_u_k,self.M)
        return np.array(X_vector), np.array(u_vector)
    
    def simulate_linear(self, X0, t_samples, trajectory, u_eq):
        #u_minus_1 = np.array(u_eq) # TODO: Confirmar se é u_eq ou 0
        u_minus_1 = np.zeros(4)
        x_k = X0
        u_k_minus_1 = u_minus_1
        X_vector = [X0]
        u_vector = []
        Hqp = self.Hqp
        cvxopt.solvers.options['show_progress'] = False

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            #ref_N = trajectory[k:k+self.N] # TODO validar se termina em k+N-1 ou em k+N
            ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = self.phi @ epsilon_k
            fqp = 2*np.transpose(self.Gn) @ (f - ref_N)
            
            # bqp
            bqp = np.concatenate((
                np.tile(self.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(self.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(self.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - self.restrictions['u_min'], self.M),
                np.tile(self.restrictions['y_max'], self.N) - f,
                f - np.tile(self.restrictions['y_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            self.Aqp = self.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(self.Aqp), cvxopt.matrix(bqp))
            x = np.array(res['x']).reshape((Hqp.shape[1],))
            ##################################################


            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(4), np.zeros((4, 4*(self.M - 1)))), axis = 1) @ x # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq

            # Apply control u_k in the multi-rotor
            x_k = self.Ad @ x_k + self.Bd @ u_k
            #if np.linalg.norm(x_k[9:12]) > 10 or np.max(np.abs(x_k[0:2])) > 1.4:
            #    print('Simulation exploded.')
            #    print('x_k =',x_k)
            #    break
            
            X_vector.append(x_k)
            u_k_minus_1 = u_k
            u_vector.append(u_k)
            #delta_u_initial = np.tile(delta_u_k,self.M)
        return np.array(X_vector), np.array(u_vector)

    def simulate_linear2(self, X0, t_samples, trajectory, u_eq):

        u_minus_1 = np.array(u_eq) # TODO: Confirmar se é u_eq ou 0
        #u_minus_1 = np.zeros(4)
        x_k = X0
        u_k_minus_1 = u_minus_1
        delta_u_initial = 0*np.ones(4*self.M)
        X_vector = [X0]
        u_vector = []
        Hqp = self.Hqp
        cvxopt.solvers.options['show_progress'] = False
        linear_sys = StateSpace(self.Ad, self.Bd, self.C, np.zeros((3,4)))

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            #ref_N = trajectory[k:k+self.N] # TODO validar se termina em k+N-1 ou em k+N
            ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = self.phi @ epsilon_k
            fqp = 2*np.transpose(self.Gn) @ (f - ref_N)
            
            # bqp
            bqp = np.concatenate((
                np.tile(self.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(self.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(self.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - self.restrictions['u_min'], self.M),
                np.tile(self.restrictions['y_max'], self.N) - f,
                f - np.tile(self.restrictions['y_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            self.Aqp = self.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(self.Aqp), cvxopt.matrix(bqp))
            x = np.array(res['x']).reshape((Hqp.shape[1],))
            ##################################################


            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(4), np.zeros((4, 4*(self.M - 1)))), axis = 1) @ x # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq

            # Apply control u_k in the multi-rotor
            t_simulation = np.arange(t_samples[k], t_samples[k+1], self.T)
            #t_simulation2 = np.arange(0, self.T_sample, self.T)

            #u_k_vector = np.array([
            #    u_k[0]*np.ones(len(t_simulation)),
            #    u_k[1]*np.ones(len(t_simulation)),
            #    u_k[2]*np.ones(len(t_simulation)),
            #    u_k[3]*np.ones(len(t_simulation))
            #]).transpose()

            u_k_vector = np.array([
               (u_k[0] - u_eq[0])*np.ones(2),
               (u_k[1] - u_eq[1])*np.ones(2),
               (u_k[2] - u_eq[2])*np.ones(2),
               (u_k[3] - u_eq[3])*np.ones(2)
            ]).transpose()

            #_, _, x_k = lsim(linear_sys, u_k_vector, t_simulation, X0 = x_k)
            _, _, x_k = lsim(linear_sys, u_k_vector, [t_samples[k], t_samples[k+1]], X0 = x_k)
            x_k = x_k[-1]
            #if np.linalg.norm(x_k[9:12]) > 10 or np.max(np.abs(x_k[0:2])) > 1.4:
            #    print('Simulation exploded.')
            #    print('x_k =',x_k)
            #    break
            
            X_vector.append(x_k)
            u_k_minus_1 = u_k
            u_vector.append(u_k)
            #delta_u_initial = np.tile(delta_u_k,self.M)
        return np.array(X_vector), np.array(u_vector)
    
    def simulate_linear3(self, X0, t_samples, trajectory, u_eq):
        u_minus_1 = np.array(u_eq) # TODO: Confirmar se é u_eq ou 0
        #u_minus_1 = np.zeros(4)
        x_k = X0
        u_k_minus_1 = u_minus_1
        X_vector = [X0]
        u_vector = []
        Hqp = self.Hqp
        cvxopt.solvers.options['show_progress'] = False

        for k in range(0, len(t_samples)-1): # TODO: confirmar se é -1 mesmo:
            #ref_N = trajectory[k:k+self.N] # TODO validar se termina em k+N-1 ou em k+N
            ref_N = np.tile(trajectory[k,:], self.N)
            epsilon_k = np.concatenate((x_k, u_k_minus_1), axis = 0)
            f = self.phi @ epsilon_k
            fqp = 2*np.transpose(self.Gn) @ (f - ref_N)
            
            # bqp
            bqp = np.concatenate((
                np.tile(self.restrictions['delta_u_max'], self.M), # delta_u_max_M
                - np.tile(self.restrictions['delta_u_min'], self.M), # -delta_u_min_M
                np.tile(self.restrictions['u_max'] - u_k_minus_1, self.M),
                np.tile(u_k_minus_1 - self.restrictions['u_min'], self.M),
                np.tile(self.restrictions['y_max'], self.N) - f,
                f - np.tile(self.restrictions['y_min'], self.N)
            ), axis = 0)

            # optimization
            #cost_function = lambda x: 0.5*np.transpose(x) @ Hqp @ x + np.transpose(fqp) @ x

            #constraints_dict = {
            #    'type': 'ineq',
            #    'fun': lambda x: -(self.Aqp @ x - bqp)
            #}
            #opt = {'maxiter': 1000}
            #res = minimize(fun= cost_function, x0=delta_u_initial, constraints=constraints_dict)#, options=opt) ######################### TODO: verificar opt ######################
            #print('Hqp',np.max(np.abs(Hqp - Hqp.T)))
           
            # METODO 2 #######################################
            Hqp = Hqp.astype(np.double)
            fqp = fqp.astype(np.double)
            self.Aqp = self.Aqp.astype(np.double)
            bqp = bqp.astype(np.double)
            res = cvxopt.solvers.qp(cvxopt.matrix(Hqp), cvxopt.matrix(fqp), cvxopt.matrix(self.Aqp), cvxopt.matrix(bqp))
            x = np.array(res['x']).reshape((Hqp.shape[1],))
            ##################################################


            #print('res.x',res.x)
            #print('res2.x',np.array(res2['x']).reshape((Hqp.shape[1],)))
            delta_u_k = np.concatenate((np.eye(4), np.zeros((4, 4*(self.M - 1)))), axis = 1) @ x # optimal delta_u_k
            u_k = u_k_minus_1 + delta_u_k # TODO: confirmar se tem esse u_eq

            # Apply control u_k in the multi-rotor
            x_k = self.Ad @ x_k + self.Bd @ u_k
            #if np.linalg.norm(x_k[9:12]) > 10 or np.max(np.abs(x_k[0:2])) > 1.4:
            #    print('Simulation exploded.')
            #    print('x_k =',x_k)
            #    break
            
            X_vector.append(x_k)
            u_k_minus_1 = u_k
            u_vector.append(u_k)
            #delta_u_initial = np.tile(delta_u_k,self.M)
        return np.array(X_vector), np.array(u_vector)

    
    def discretize(self):
        #sys = StateSpace(self.A,self.B,self.C, np.zeros((3,4)))
        sys_d = cont2discrete((self.A,self.B,self.C, np.zeros((3,4))), self.T_sample, 'zoh')
        print('sys_d',sys_d)
        Ad, Bd, Cd, _, _ = sys_d
        return Ad, Bd, Cd