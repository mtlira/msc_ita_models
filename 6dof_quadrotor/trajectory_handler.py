import numpy as np
from parameters.simulation_parameters import T_sample

# x, y and z coordinates are on the upside-down body axis

class TrajectoryHandler(object):
    def __init__(self):
        pass

    def point(self, xp, yp, zp, T_simulation, include_psi = True):
        t = np.arange(0, T_simulation, T_sample)
        r_point = np.array([xp*np.ones(len(t)),
                    yp*np.ones(len(t)),
                    zp*np.ones(len(t)),
                    0*t,
                    0*t,
                    0*t
                    ]).transpose()
        
        if not include_psi:
            r_point = r_point[:, :-1]

        return r_point
    
    def line(self, a, b, c, clamp, T_simulation):
        """
        Creates a line trajectory defined by [a.t b.t c.t]
        clamp: maximum coordinate value allowed for the trajectory
        """
        t = np.arange(0, T_simulation, T_sample)
        r_line = np.array([a*t, b*t, c*t, 0*t, 0*t, 0*t]).transpose()
        r_line = r_line.clip(min=-clamp, max=clamp)

        return r_line            
    
    def circle_xy(self, w, r, T_simulation, include_psi = True):
        t = np.arange(0, T_simulation, T_sample)
        r_circle_xy = np.array([r*np.sin(w*t),
                       (r - r*np.cos(w*t)),
                       np.zeros(len(t)),
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()
        
        if not include_psi:
            r_circle_xy = r_circle_xy[:, :-1]

        return r_circle_xy
    
    def circle_xz(self, w, r, T_simulation, include_psi = True):
        t = np.arange(0, T_simulation, T_sample)
        r_circle_xz = np.array([r*np.sin(w*t),
                        np.zeros(len(t)),
                        (r - r*np.cos(w*t)),
                        0*t,
                        0*t,
                        0*t
                       ]).transpose()
        
        if not include_psi:
            r_circle_xz = r_circle_xz[:, :-1]

        return r_circle_xz
    
    def lissajous_xy(self, w, r, T_simulation, include_psi = True):
        t = np.arange(0, T_simulation, T_sample)
        r_lissajous_xy = np.array([r*np.sin(-w*t + np.pi/2) - r,
                       (r*np.sin(-2/3*w*t)),
                       np.zeros(len(t)),
                       0*t,
                       0*t,
                       0*t
                       ]).transpose()    
        
        if not include_psi:
            r_lissajous_xy = r_lissajous_xy[:, :-1]    

        return r_lissajous_xy
    
    def helicoidal(self, w, T_simulation):
        t = np.arange(0, T_simulation, T_sample)
        r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       -1*t,
                       ]).transpose()

        return r_helicoidal

    def helicoidal_znegative(self, w, T_simulation):
        t = np.arange(0, T_simulation, T_sample)
        r_helicoidal = np.array([5*(1 + 0.1*t)*np.sin(w*t),
                       (5 - 5*(1 + 0.1*t)*np.cos(w*t)),
                       1*t,
                       ]).transpose()
    
        
        return r_helicoidal
    
    def generate_trajectory(self, trajectory_type, args):
        if trajectory_type == 'point':
            return self.point(args[0], args[1], args[2], args[3])
        
        if trajectory_type == 'line':
            return self.line(args[0], args[1], args[2], args[3], args[4])
        
        if trajectory_type == 'circle_xy':
            return self.circle_xy(args[0], args[1], args[2])
        
        if trajectory_type == 'lissajous_xy':
            return self.lissajous_xy(args[0], args[1], args[2])
        
        if trajectory_type == 'circle_xz':
            return self.circle_xz(args[0], args[1], args[2])
        
        if trajectory_type == 'helicoidal':
            return self.helicoidal(args[0], args[1])
        
        if trajectory_type == 'helicoidal_znegative':
            return self.helicoidal_znegative(args[0], args[1])
        
        raise ValueError('Trajectory type not compatible')
    
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

    def generate_point_trajectories(self, point_numbers, T_simulation):
        points_vector = []

        points_vector.append(np.array([0.0,0.0,0.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,1.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-1.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,2.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-2.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,5.0, T_simulation]))
        points_vector.append(np.array([0.0,0.0,-5.0, T_simulation]))
        points_vector.append(np.array([1.0,0.0,0.0, T_simulation]))
        points_vector.append(np.array([-1.0,0.0,0.0, T_simulation]))
        points_vector.append(np.array([0.0,1.0,0.0, T_simulation]))
        points_vector.append(np.array([0.0,-1.0,0.0, T_simulation]))

        # Random points
        for i in range(point_numbers - len(points_vector)):
            point = 6*np.random.rand(3) - 3 # Random point inside sphere of 3m radius centered in (0,0,0)
            point = np.concatenate((point, [T_simulation]), axis = 0)
            points_vector.append(point)
        
        return points_vector
    
    def generate_circle_xy_trajectories(self):
        short_radius_vector = np.arange(0.5, 5, 0.5)
        long_radius_vector = np.arange(5, 10, 0.2)
        short_period_vector = np.arange(2, 10, 1)
        long_period_vector = np.arange(10, 30, 1)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 1.25*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)

        return args
    
    def generate_circle_xz_trajectories(self):
        short_radius_vector = np.arange(1, 5, 1)
        long_radius_vector = np.arange(5, 10, 1)
        short_period_vector = np.arange(2, 8, 1)
        long_period_vector = np.arange(10, 20, 1)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 1.25*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        print('num circles xz =',num_circles)
        return args


    def generate_line_trajectories(self, num_lines):
        coefficients = 8*np.random.rand(num_lines, 3) - 4
        clamp = 5*np.random.rand(num_lines, 1) + 10
        T_simulation = 25*np.ones((num_lines, 1))

        args = np.concatenate((coefficients, clamp, T_simulation), axis = 1)
        return args
    
    def generate_lissajous_xy_trajectories(self):
        short_radius_vector = np.arange(1, 5, 1)
        long_radius_vector = np.arange(5, 10, 1)
        short_period_vector = np.arange(4, 8, 1)
        long_period_vector = np.arange(10, 20, 1)

        w_short_vector = 2*np.pi/short_period_vector
        w_long_vector = 2*np.pi/long_period_vector

        args = []

        for period in short_period_vector:
            for radius in short_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        for period in long_period_vector:
            for radius in long_radius_vector:
                #args = np.concatenate((args, [[2*np.pi/period, radius, 1.3*period]]), axis = 0)
                args.append([2*np.pi/period, radius, 3*period])
        num_circles = len(short_radius_vector) * len(short_period_vector) + len(long_radius_vector) * len(long_period_vector)
        print('lissajous_xy trajectories =',num_circles)
        return args

teste = TrajectoryHandler()
teste.generate_circle_xy_trajectories()
#args = teste.generate_line_trajectories()
#print('argsshape',np.shape(args))

