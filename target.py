import numpy as np


class Target:
    def __init__(self,
                 init_weight=1.0,
                 init_state=np.array([[0.0], [0.0], [0.0], [0.0]]),
                 init_cov=np.diag((0.01, 0.01, 0.01, 0.01)),
                 process_noise=0.001,
                 step=3,
                 dt_1=1,
                 dt_2=1):
        self.state = init_state
        self.state_cov = init_cov
        self.weight = init_weight
        self.measure_cov = init_cov
        self.dt_1 = dt_1
        self.dt_2 = dt_2

        self.all_states = []
        self.all_states.append(init_state)

        self.all_cov = []
        self.all_cov.append(init_cov)

        self.state[2][0] = step
        self.state[3][0] = step
        self.A = np.array([[1, 0, dt_1, 0],
                           [0, 1, 0, dt_2],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.eye(init_state.shape[0])
        self.U = np.zeros((init_state.shape[0], 1))

        self.Q = np.eye(init_state.shape[0])

        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = np.eye(2)

        self.process_noise = process_noise

    def set_dir(self, dt_1, dt_2):
        self.A = np.array([[1, 0, dt_1, 0],
                           [0, 1, 0, dt_2],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

    def next_state(self, noise=False):
        x = self.state
        next_state = np.dot(self.A, x) + np.dot(self.B, self.U)

        # Add small process noise
        if noise:
            Qsim = np.diag([self.process_noise, self.process_noise]) ** 2
            next_state[0, 0] = next_state[0, 0] + np.random.randn() * Qsim[0, 0]
            next_state[1, 0] = next_state[1, 0] + np.random.randn() * Qsim[1, 1]
        self.state = next_state
        self.all_states.append(next_state)

        next_cov = np.dot(self.A, np.dot(self.state_cov, self.A.T)) + self.Q
        self.state_cov = next_cov
        self.all_cov.append(next_cov)

    def get_measurement(self):
        obs = np.dot(self.H, self.state)
        self.measure_cov = self.R + np.dot(self.H,
                                           np.dot(self.state_cov, self.H.T))

        return obs

    def sample(self, N=1):
        return np.random.multivariate_normal(self.state.flat, self.state_cov, size=N)
