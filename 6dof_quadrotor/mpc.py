import numpy as np
import pandas as pd

class mpc(object):
    def __init__(self, M, N, rho, A, B, C, time_step, output_weights, control_weights):
        self.M = M
        self.N = N
        self.rho = rho
        self.A = A
        self.B = B
        self.C = C
        self.T = time_step
        self.output_weights = output_weights
        self.control_weights = control_weights

    def initialize_matrices(self):
        # 1. Discretization of the space state
        Ad = np.eye(np.shape(self.A)[0]) + self.A*self.T
        Bd = self.B*self.T

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
        G = np.array(column)
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
        column = Q
        for i in range(1, self.N):
            column = np.concatenate((column, np.zeros((3,3))), axis = 0)
        
        # Q - Other columns
        Q_super = column
        for i in range(1, self.N):
            column = np.roll(column, 3, axis = 0)
            column[0:3, 0:3] = np.zeros((3,3))
            Q_super = np.concatenate((Q_super, column), axis = 1)
        #print('Q_super=\n',pd.DataFrame(Q_super))
        #print('diag Q_super =',np.diag(Q_super))
        #print('shape Qsuper =',np.shape(Q_super))

        # R - First column
        column = R
        for i in range(1, self.M):
            column = np.concatenate((column, np.zeros((4,4))), axis = 0)
        
        # R - Other columns
        R_super = column
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
        
        T_M = column
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


        return A_tilda, B_tilda, C_tilda, G, Q_super