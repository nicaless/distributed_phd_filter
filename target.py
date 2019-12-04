import math
import numpy as np
from scipy.stats import multivariate_normal as mv_normal

class Target:
    def __init__(self,
                 init_state,
                 measurement_model=mv_normal(),
                 process_noise=0.001,
                 step=3):
        self.init_state = init_state

        self.all_states = []
        self.all_states.append(init_state)

        self.A = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 0]])
        self.B = np.array([[math.cos(0), 0],
                           [math.sin(0), 0],
                           [0.0, 1],
                           [1.0, 0.0]])
        self.U = np.array([[step, 0.1]]).T

        self.process_noise = process_noise

    def next_state(self):
        x = self.init_state
        self.B[0, 0] = 0.1 * math.cos(x[2, 0])
        self.B[1, 0] = 0.1 * math.sin(x[2, 0])
        next_state = np.dot(self.A, x) + np.dot(self.B, self.U)

        # Add small process noise
        Qsim = np.diag([self.process_noise, self.process_noise]) ** 2
        next_state[0, 0] = next_state[0, 0] + np.random.randn() * Qsim[0, 0]
        next_state[1, 0] = next_state[1, 0] + np.random.randn() * Qsim[1, 1]
        self.init_state = next_state
        self.all_states.append(next_state)

    def get_measurement(self):
        return current_pos + mv_normal

        measures[n] = current_pos + normal(0,
                                           self.sensor_noise[n],
                                           current_pos.shape)
