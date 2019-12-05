import math
import numpy as np


class Target:
    def __init__(self,
                 init_state=np.array([[0.0], [0.0], [0.0], [0.0]]),
                 process_noise=0.001,
                 step=3):
        self.state = init_state

        self.all_states = []
        self.all_states.append(init_state)

        self.A = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 0]])
        self.B = np.array([[math.cos(0), 0],
                           [math.sin(0), 0],
                           [0.0, 1],
                           [1.0, 0.0]])
        self.U = np.array([[step, 0.1]]).T

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.process_noise = process_noise

    def next_state(self, x=None):
        if x is None:
            x = self.state
        self.B[0, 0] = 0.1 * math.cos(x[2, 0])
        self.B[1, 0] = 0.1 * math.sin(x[2, 0])
        next_state = np.dot(self.A, x) + np.dot(self.B, self.U)

        # Add small process noise
        Qsim = np.diag([self.process_noise, self.process_noise]) ** 2
        next_state[0, 0] = next_state[0, 0] + np.random.randn() * Qsim[0, 0]
        next_state[1, 0] = next_state[1, 0] + np.random.randn() * Qsim[1, 1]
        self.state = next_state
        self.all_states.append(next_state)
        if x is not None:
            return next_state

    def get_measurement(self):
        current_pos = np.dot(self.H, self.state)
        return current_pos + np.random.normal(0, 1, current_pos.shape)
