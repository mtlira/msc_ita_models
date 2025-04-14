import numpy as np

# x, y and z coordinates are on the upside-down body axis

class TrajectoryHandler(object):
    def __init__(self):
        pass

    def point(self, xp, yp, zp, t, include_psi):
        r_point = np.array([xp*np.ones(len(t)),
                    yp*np.ones(len(t)),
                    zp*np.ones(len(t)),
                    ]).transpose()
        
        if include_psi:
            r_point = np.concatenate((r_point, np.array([np.zeros(len(t))]).transpose()), axis = 1)
        
        return r_point
    
    def point_phibeta(self, xp, yp, zp, t, include_psi):
        r_point = np.array([xp*np.ones(len(t)),
                    yp*np.ones(len(t)),
                    zp*np.ones(len(t)),
                    0*t,
                    0*t
                    ]).transpose()
                
        return r_point
    
    def line(self, a, b, c, t, clamp=None, include_psi = False):
        """
        Creates a line trajectory defined by [a.t b.t c.t]
        clamp: maximum coordinate value allowed for the trajectory
        """
        r_line = np.array([a*t, b*t, c*t]).transpose()
        if clamp is not None:
            r_line = r_line.clip(min=-clamp, max=clamp)

        if include_psi:
            r_line = np.concatenate((r_line, np.array([np.zeros(len(t))]).transpose()), axis = 1)

        return r_line            

    def circle_xy(self, w, r, t, include_psi = False):
        r_circle_xy = np.array([r*np.sin(w*t),
                       (r - r*np.cos(w*t)),
                       np.zeros(len(t)),
                       ]).transpose()
        
        if include_psi:
            r_circle_xy = np.concatenate((r_circle_xy, np.array([np.zeros(len(t))]).transpose()), axis = 1)
        return r_circle_xy
    
    def circle_xy_phibetapsi(self, w, r, t, include_psi = False):
        r_circle_xy = np.array([r*np.sin(w*t),
                       (r - r*np.cos(w*t)),
                       np.zeros(len(t)),
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()
        
        return r_circle_xy
    
    def circle_xz(self, w, r, t, include_psi = False):
        r_circle_xz = np.array([r*np.sin(w*t),
                       np.zeros(len(t)),
                       (r - r*np.cos(w*t)),
                       ]).transpose()
        
        if include_psi:
            r_circle_xz = np.concatenate((r_circle_xz, np.array([np.zeros(len(t))]).transpose()), axis = 1)

        return r_circle_xz

    
    def helicoidal(self, w, t, include_psi = False):
        r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       -1*t,
                       ]).transpose()

        if include_psi:
            r_helicoidal = np.concatenate((r_helicoidal, np.array([np.zeros(len(t))]).transpose()), axis = 1)

        return r_helicoidal

    def helicoidal_znegative(self, w, t, include_psi=False):
        r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       1*t,
                       ]).transpose()
        
        if include_psi:
            r_helicoidal = np.concatenate((r_helicoidal, np.array([np.zeros(len(t))]).transpose()), axis = 1)
        
        return r_helicoidal
    
    def speed_reference(self, trajectory, t):
        '''
        Calculates the speed components in the Earth reference by derivating the trajectory [x(t) y(t) z(t)] vector. Returns the trajectory vector with the speed references concatenated.
        '''
        speed_ref = [(trajectory[1] - trajectory[0])/(t[1] - t[0])]
        for i in range(1, len(trajectory) - 1):
            drdt = (trajectory[i+1] - trajectory[i-1])/(t[i+1] - t[i-1])
            speed_ref.append(drdt)
        speed_ref.append(speed_ref[-1]) # Repete penultima derivada #TODO: verificar uma forma de melhoria de codigo (prever futuro, etc)
        speed_ref = np.array(speed_ref)
        tr = np.concatenate((speed_ref, trajectory), axis = 1)
        return tr

    def generate_point_trajectories(self, point_numbers):
        points_vector = []

        points_vector.append(np.array([0.0,0.0,0.0]))
        points_vector.append(np.array([0.0,0.0,1.0]))
        points_vector.append(np.array([0.0,0.0,-1.0]))
        points_vector.append(np.array([0.0,0.0,2.0]))
        points_vector.append(np.array([0.0,0.0,-2.0]))
        points_vector.append(np.array([0.0,0.0,5.0]))
        points_vector.append(np.array([0.0,0.0,-5.0]))
        points_vector.append(np.array([1.0,0.0,0.0]))
        points_vector.append(np.array([-1.0,0.0,0.0]))
        points_vector.append(np.array([0.0,1.0,0.0]))
        points_vector.append(np.array([0.0,-1.0,0.0]))

        # Random points
        for i in range(point_numbers - len(points_vector)):
            point = 6*np.random.rand(3) - 3 # Random point inside sphere of 20m radius centered in (0,0,0)
            points_vector.append(point)
        
        return points_vector
