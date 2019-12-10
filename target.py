import math
import numpy as np


class Target:
    def __init__(self,
                 init_weight=1.0,
                 init_state=np.array([[0.0], [0.0], [0.0], [0.0]]),
                 init_cov=np.diag((0.01, 0.01, 0.01, 0.01)),
                 process_noise=0.001,
                 step=3):
        self.state = init_state
        self.cov = init_cov
        self.weight = init_weight

        self.all_states = []
        self.all_states.append(init_state)

        self.all_cov = []
        self.all_cov.append(init_cov)

        self.A = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 0]])
        self.B = np.array([[math.cos(0), 0],
                           [math.sin(0), 0],
                           [0.0, 1],
                           [1.0, 0.0]])
        self.U = np.array([[step, 0.1]]).T
        self.Q = np.eye(init_state.shape[0])

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        self.process_noise = process_noise

    def next_state(self, noise=False):
        x = self.state
        self.B[0, 0] = 0.1 * math.cos(x[2, 0])
        self.B[1, 0] = 0.1 * math.sin(x[2, 0])
        next_state = np.dot(self.A, x) + np.dot(self.B, self.U)

        # Add small process noise
        if noise:
            Qsim = np.diag([self.process_noise, self.process_noise]) ** 2
            next_state[0, 0] = next_state[0, 0] + np.random.randn() * Qsim[0, 0]
            next_state[1, 0] = next_state[1, 0] + np.random.randn() * Qsim[1, 1]
        self.state = next_state
        self.all_states.append(next_state)

        next_cov = np.dot(self.A, np.dot(self.cov, self.A.T)) + self.Q
        self.cov = next_cov
        self.all_cov.append(next_cov)

    def get_measurement(self):
        obs = np.dot(self.H, self.state)
        return obs

    def sample(self, N=1):
        return np.random.multivariate_normal(self.state.flat, self.cov, size=N)


