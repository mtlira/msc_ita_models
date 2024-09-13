class PID(object):
    def __init__(self, KP, KI, KD, setpoint, time_step):
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.setpoint = setpoint
        self.error = 0
        self.proportional_error = 0
        self.previous_error = 0
        self.integral_error = 0
        self.derivative_error = 0
        self.output = 0
        self.time_step = time_step

    def compute(self, feedback):
        self.error = self.setpoint - feedback
        self.proportional_error = self.KP*self.error
        self.integral_error += self.KI*self.error*self.time_step
        self.derivative_error = self.KD*(self.error - self.previous_error)/self.time_step
        self.previous_error = self.error
        self.output = self.proportional_error + self.integral_error + self.derivative_error
        return self.output

