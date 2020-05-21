import numpy as np


dt_1 = 0
dt_2 = 0
DEFAULT_INIT_STATE = np.array([[0.0], [0.0], [0.0], [0.0]])
DEFAULT_INIT_COV = np.diag((0.01, 0.01, 0.01, 0.01))
DEFAULT_A = np.array([[1, 0, dt_1, 0],
                      [0, 1, 0, dt_2],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
DEFAULT_B = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 1]])
DEFAULT_H = np.eye(DEFAULT_INIT_STATE.shape[0])


class Target:
    def __init__(self,
                 init_weight=1.0,
                 init_state=DEFAULT_INIT_STATE,
                 init_cov=DEFAULT_INIT_COV,
                 A=DEFAULT_A,
                 B=DEFAULT_B,
                 H=DEFAULT_H,
                 process_noise=0.001):
        self.state = init_state
        self.state_cov = init_cov
        self.weight = init_weight
        self.measure_cov = init_cov

        self.A = A
        self.B = B
        self.U = np.zeros(B.shape[1])

        self.Q = np.eye(init_state.shape[0])

        self.H = H
        self.R = np.eye(H.shape[0])

        self.process_noise = process_noise

        self.all_states = []
        self.all_states.append(init_state)

        self.all_cov = []
        self.all_cov.append(init_cov)

    def set_state_cov(self, state, cov):
        self.state = state
        self.state_cov = cov

        self.all_states.append(state)
        self.all_cov.append(cov)

    def next_state(self, input=None, noise=False):
        x = self.state
        if input is not None:
            next_state = np.dot(self.A, x) + np.dot(self.B, input)
        else:
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

    def get_measurement(self, R=None):
        if R is None:
            R = self.R
        obs = np.dot(self.H, self.state)
        self.measure_cov = R + np.dot(self.H, np.dot(self.state_cov, self.H.T))

        return obs

    def sample(self, N=1):
        return np.random.multivariate_normal(self.state.flat, self.state_cov, size=N)



