import numpy as np
import sympy as sp

# x = [  phi (0)
#        theta (1)
#        psi (2)
#        p (3)
#        q (4)
#        r (5)
#        u (6)
#        v (7)
#        w (8)
#        x (9)
#        y (10)
#        z (11)
#      ]

#u = [
#    ft (0) # f_t está no eixo do corpo
#    tx (1)
#    ty (2)
#    tz (3)
#]

class multirotor(object):
    def __init__(self, m, g, I_x, I_y, I_z, b, l, d, num_rotors):
        self.m = m
        self.g = g
        self.I_x = I_x
        self.I_y = I_y
        self.I_z = I_z
        self.b = b
        self.l = l
        self.d = d
        if num_rotors not in [4, 8]:
            raise ValueError('Valid numbers of rotors are 4 and 8 only.')
        self.num_rotors = num_rotors
        # [u] = Gama * [w]
        if num_rotors == 4:
            self.Gama = np.array([[b, b, b, b],
                         [-b*l, 0, b*l, 0],
                         [0, b*l, 0, -b*l],
                         [d, -d, d, -d]])
            
        if num_rotors == 8:

            sq2o2 = np.sin(np.pi/4)
            self.Gama = np.array([
                [b, b, b, b, b, b, b, b],
                [b*l, b*l*sq2o2, 0, -b*l*sq2o2, -b*l, -b*l*sq2o2, 0, b*l*sq2o2],
                [0, b*l*sq2o2, b*l, b*l*sq2o2, 0, -b*l*sq2o2, -b*l, -b*l*sq2o2],
                [d, -d, d, -d, d, -d, d, -d]
            ])

    # State-space model functions

    # Para testar u como argumento (PID)
    def f2(self, X, t, f_t, t_x, t_y, t_z):
        m = self.m
        g = self.g
        I_x = self.I_x
        I_y = self.I_y
        I_z = self.I_z

        phi, theta, psi, p, q, r, u, v, w, x, y, z = X
        dx_dt = [
        p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
        q*np.cos(phi) - r*np.sin(phi),
        r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
        (I_y - I_z)/I_x * r*q + t_x/I_x,
        (I_z - I_x)/I_y * p*r + t_y/I_y,
        (I_x - I_y)/I_z * p*q + t_z/I_z,
        r*v - q*w - g*np.sin(theta),
        p*w - r*u + g*np.sin(phi)*np.cos(theta),
        q*u - p*v + g*np.cos(theta)*np.cos(phi) - f_t/m, # Sabatino é - ft/m!!!!!
        w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - v*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + u*(np.cos(psi)*np.cos(theta)),
        v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - w*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)) + u*(np.cos(theta)*np.sin(psi)),
        w*(np.cos(phi)*np.cos(theta)) - u*(np.sin(theta)) + v*(np.cos(theta)*np.sin(phi))
        ]
        return dx_dt

    def f2_bresciani(self, X, t, f_t, t_x, t_y, t_z):
        m = self.m
        g = self.g
        I_x = self.I_x
        I_y = self.I_y
        I_z = self.I_z

        phi, theta, psi, p, q, r, u, v, w, x, y, z = X
        dx_dt = [
        p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
        q*np.cos(phi) - r*np.sin(phi),
        r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
        (I_y - I_z)/I_x * r*q + t_x/I_x,
        (I_z - I_x)/I_y * p*r + t_y/I_y,
        (I_x - I_y)/I_z * p*q + t_z/I_z,
        r*v - q*w + g*np.sin(theta),
        p*w - r*u - g*np.sin(phi)*np.cos(theta),
        q*u - p*v - g*np.cos(theta)*np.sin(phi) + f_t/m,
        w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - v*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + u*(np.cos(psi)*np.cos(theta)),
        v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - w*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)) + u*(np.cos(theta)*np.sin(psi)),
        w*(np.cos(phi)*np.cos(theta)) - u*(np.sin(theta)) + v*(np.cos(theta)*np.sin(phi))
        ]
        return dx_dt

    # def f(self, X, t):
    #     m = self.m
    #     g = self.g
    #     I_x = self.I_x
    #     I_y = self.I_y
    #     I_z = self.I_z

    #     phi, theta, psi, p, q, r, u, v, w, x, y, z = X
    #     f_t, t_x, t_y, t_z = u_sim(t)
    #     dx_dt = [
    #     p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
    #     q*np.cos(phi) - r*np.sin(phi),
    #     r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
    #     (I_y - I_z)/I_x * r*q + t_x/I_x,
    #     (I_z - I_x)/I_y * p*r + t_y/I_y,
    #     (I_x - I_y)/I_z * p*q + t_z/I_z,
    #     r*v - q*w - g*np.sin(theta),
    #     p*w - r*u + g*np.sin(phi)*np.cos(theta),
    #     q*u - p*v + g*np.cos(theta)*np.cos(phi) - f_t/m,
    #     w*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)) - v*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + u*(np.cos(psi)*np.cos(theta)),
    #     v*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - w*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)) + u*(np.cos(theta)*np.sin(psi)),
    #     w*(np.cos(phi)*np.cos(theta)) - u*(np.sin(theta)) + v*(np.cos(theta)*np.sin(phi))
    #     ]
    #     return dx_dt

    # Para achar o ponto de equilíbrio
    # def f_solve(self, X):
    #     m = self.m
    #     g = self.g
    #     I_x = self.I_x
    #     I_y = self.I_y
    #     I_z = self.I_z

    #     phi, theta, psi, p, q, r, u, v, w = X[0:9]
    #     f_t, t_x, t_y, t_z = u_eq(0)
    #     dx_dt = [
    #     p + r*np.cos(phi)*np.tan(theta) + q*np.sin(phi)*np.tan(theta),
    #     q*np.cos(phi) - r*np.sin(phi),
    #     r*np.cos(phi)/np.cos(theta) + q*np.sin(phi)/np.cos(theta),
    #     (I_y - I_z)/I_x * r*q + t_x/I_x,
    #     (I_z - I_x)/I_y * p*r + t_y/I_y,
    #     (I_x - I_y)/I_z * p*q + t_z/I_z,
    #     r*v - q*w - g*np.sin(theta),
    #     p*w - r*u + g*np.sin(phi)*np.cos(theta),
    #     q*u - p*v + g*np.cos(theta)*np.cos(phi) - f_t/m
    #     ]
    #     return dx_dt

    # Para linearização
    def f_sym(self):
        m = self.m
        g = self.g
        I_x = self.I_x
        I_y = self.I_y
        I_z = self.I_z

        X_symbols = sp.symbols('phi theta psi p q r u v w x y z')
        phi, theta, psi, p, q, r, u, v, w, x, y, z = X_symbols
        U_symbols = sp.symbols('f_t t_x t_y t_z')
        f_t, t_x, t_y, t_z = U_symbols
        
        dx_dt = [
        p + r*sp.cos(phi)*sp.tan(theta) + q*sp.sin(phi)*sp.tan(theta),
        q*sp.cos(phi) - r*sp.sin(phi),
        r*sp.cos(phi)/sp.cos(theta) + q*sp.sin(phi)/sp.cos(theta),
        (I_y - I_z)/I_x * r*q + t_x/I_x,
        (I_z - I_x)/I_y * p*r + t_y/I_y,
        (I_x - I_y)/I_z * p*q + t_z/I_z,
        r*v - q*w - g*sp.sin(theta),
        p*w - r*u + g*sp.sin(phi)*sp.cos(theta),
        q*u - p*v + g*sp.cos(theta)*sp.cos(phi) - f_t/m,
        w*(sp.sin(phi)*sp.sin(psi) + sp.cos(phi)*sp.cos(psi)*sp.sin(theta)) - v*(sp.cos(phi)*sp.sin(psi) - sp.cos(psi)*sp.sin(phi)*sp.sin(theta)) + u*(sp.cos(psi)*sp.cos(theta)),
        v*(sp.cos(phi)*sp.cos(psi) + sp.sin(phi)*sp.sin(psi)*sp.sin(theta)) - w*(sp.cos(psi)*sp.sin(phi) - sp.cos(phi)*sp.sin(psi)*sp.sin(theta)) + u*(sp.cos(theta)*sp.sin(psi)),
        w*(sp.cos(phi)*sp.cos(theta)) - u*(sp.sin(theta)) + v*(sp.cos(theta)*sp.sin(phi))
        ]
        return X_symbols, U_symbols, dx_dt
    
    def get_controls(self, w_vector):
        # [u] = Gama * [w] 
        u = self.Gama @ w_vector**2
        return u
    
    def get_omegas(self, u):
        #w_vector = np.sqrt(np.linalg.pinv(self.Gama) @ u)
        w_vector = np.sqrt(self.m*self.g/(self.num_rotors*self.b))*np.ones(self.num_rotors)
        return w_vector