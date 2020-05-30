from copy import deepcopy
import math
from numba import jit, njit, cuda
import numpy as np
from operator import attrgetter
from scipy.spatial.distance import mahalanobis
from scipy.linalg import block_diag, orth

from target import Target, DEFAULT_H


class DKFNode:
    def __init__(self,
                 node_id,
                 targets,
                 position=np.array([0, 0, 0]),
                 region=[(-50, 50), (-50, 50)],
                 R_shape=DEFAULT_H.shape[0]
                 ):
        self.node_id = node_id

        self.position = position
        self.region = region
        self.R = np.eye(R_shape)
        self.fov = (region[0][1] - region[0][0]) / 2.0
        self.detection_probability = 1

        self.gamma = 1

        self.full_state = None
        self.full_cov = None
        self.targets = targets

        # prediction results
        self.full_state_prediction = None
        self.full_cov_prediction = None
        self.predicted_pos = []
        self.predicted_targets = []

        # update results
        self.full_cov_update = None
        self.blockH = None
        self.blockR = None
        self.blockG = None
        self.all_measurements = None
        self.observed_meas = None
        self.measurements = []

        # Consensus Filter Operators
        self.omega = None
        self.qs = None

        # Consensus Intermediate Results
        self.intermediate_cov = None

        # TRACKERS
        self.observations = {}
        self.node_positions = {}

        self.preconsensus_positions = {}
        self.preconsensus_target_covs = {}

        self.consensus_positions = {}
        self.consensus_target_covs = {}

    def update_targets(self, targets):
        self.targets = targets

    def update_position(self, new_position):
        self.position = new_position
        x = new_position[0]
        y = new_position[1]
        self.region = [(x - self.fov, x + self.fov),
                       (y - self.fov, y + self.fov)]

    def predict(self, N):
        """
        Makes prediction for all targets
        :return:
        """
        # Existing Targets
        all_targets = self.targets

        # Set R for All Targets then get measurement and create new full state array
        all_states = None
        all_covs = None
        predicted_pos = []
        for p in range(len(all_targets)):
            all_states = all_targets[p].state if all_states is None else \
                np.concatenate((all_states, all_targets[p].state))

            all_covs = all_targets[p].state_cov if all_covs is None else \
                block_diag(all_covs, all_targets[p].state_cov)

            all_targets[p].next_state()
            predicted_pos.append(all_targets[p].state)

        self.predicted_pos = predicted_pos
        self.predicted_targets = all_targets

        # Create Block Diag for A, B
        A = None
        B = None
        for p in all_targets:
            A = p.A if A is None else block_diag(A, p.A)
            B = p.B if B is None else block_diag(B, p.B)

        # Predict Full State
        full_prediction = np.dot(A, all_states)  # 42d
        Q = np.eye(all_states.shape[0])  # no noise
        full_cov = np.dot(A, np.dot(N * all_covs, A.T)) + N * Q  # 42a

        self.full_state_prediction = full_prediction
        self.full_cov_prediction = full_cov

    def get_measurements(self, targets):
        measurements = []
        for p in targets:
            new_meas = p.get_measurement(R=self.R)
            measurements.append(new_meas)
        return measurements

    def update(self, measurements):
        """
        Updates target states based on measurements
        :param measurements: array of all measurements
        :return:
        """

        all_covs = None
        all_measurements = None
        H = None
        R = None
        G = None
        observed_meas = None
        for i, m in enumerate(measurements):
            all_covs = self.predicted_targets[i].state_cov if all_covs is None \
                else np.concatenate((all_covs, self.predicted_targets[i].state_cov))
            G = self.predicted_targets[i].B if G is None else block_diag(G, self.predicted_targets[i].B)

            if self.check_measure_oob(m) or \
                            np.random.rand() > self.detection_probability or \
                            m is None:
                H = np.zeros(self.predicted_targets[i].H.shape) if H is None else \
                    block_diag(H, np.zeros(self.predicted_targets[i].H.shape))
                observed_meas = m if observed_meas is None else \
                    np.concatenate((observed_meas,
                                    self.predicted_targets[i].state))
                continue
            else:
                all_measurements = m if all_measurements is None else np.concatenate((all_measurements, m))
                H = self.predicted_targets[i].H if H is None else block_diag(H, self.predicted_targets[i].H)
                R = self.predicted_targets[i].R if R is None else block_diag(R, self.predicted_targets[i].R)
                observed_meas = m if observed_meas is None else \
                    np.concatenate((observed_meas, m))

        Hshape = H.shape
        H = H[~np.all(H == 0, axis=1)]
        if len(H) == 0:
            # TODO is this the right contingency plan?
            print("{n} made no observations".format(n=self.node_id))
            return None
        else:
            self.full_cov_update = all_covs
            self.all_measurements = all_measurements
            self.blockH = H
            self.blockR = R
            self.blockG = G
            self.observed_meas = observed_meas

        self.measurements = measurements

    def check_measure_oob(self, m):
        if m is None:
            return True

        x = m[0][0]
        y = m[1][0]

        x_out_of_bounds = x < self.region[0][0] or x > self.region[0][1]
        y_out_of_bounds = y < self.region[1][0] or y > self.region[1][1]

        return x_out_of_bounds or y_out_of_bounds

    def init_consensus(self):
        """
        initializes S (or omega) and z (or qs)
        :return:
        """
        self.omega = np.dot(self.blockH.T, np.dot(np.linalg.inv(self.blockR),
                                                  self.blockH))
        self.qs = np.dot(self.blockH.T, np.dot(np.linalg.inv(self.blockR),
                                               self.all_measurements))

    def consensus_filter(self, neighbor_omegas, neighbor_qs, neighbor_weights):
        sum_omega = self.omega
        assert len(neighbor_omegas) == len(neighbor_qs) == len(neighbor_weights)
        for i in range(len(neighbor_omegas)):
            sum_omega += np.dot(neighbor_weights[i], neighbor_omegas[i])

        sum_qs = self.qs
        for i in range(len(neighbor_omegas)):
            sum_qs += np.dot(neighbor_weights[i], neighbor_qs[i])

        self.omega = sum_omega
        self.qs = sum_qs

    def intermediate_cov_update(self):
        P = self.full_cov_prediction
        P = np.linalg.inv(np.linalg.inv(P) + self.omega)  # 42b
        self.intermediate_cov = P

    def after_consensus_update(self, N):
        P = self.intermediate_cov
        X = self.omega - np.dot(self.omega, np.dot(P, self.omega))  #42c

        SP = np.dot(self.omega, P)
        t = np.dot(np.eye(SP.shape[0]) - SP, self.qs)  # 42e

        gram = np.dot(self.blockG.T, np.dot(X, self.blockG))
        gram = gram + (np.random.random(gram.shape) * 0.001)
        d = np.dot(
                np.linalg.inv(gram),
                np.dot(self.blockG.T, t) - np.dot(self.blockG.T, np.dot(X, self.full_state_prediction)))  # 42f

        x = self.full_state_prediction + np.dot(self.blockG, d)  # 42g
        x = x + np.dot(self.full_cov_prediction, t - np.dot(X, x))  # 42h

        Y = np.dot(self.blockG,
                   np.dot(np.linalg.inv(gram), self.blockG.T))  # 42i
        P = self.full_cov_prediction - \
            np.dot(Y, np.dot(X, self.full_cov_prediction)) - \
            np.dot(self.full_cov_prediction, np.dot(X, Y))  # 42j

        PX = np.dot(self.full_cov_prediction, X)
        P = np.dot(np.eye(PX.shape[0]) - PX,
                   P - np.dot(self.full_cov_prediction, np.dot(X, Y)))  #42k

        self.full_state = x
        self.full_cov = P / N

        after_consensus_targets = self.predicted_targets
        for i, u in enumerate(after_consensus_targets):
            t_state = deepcopy(self.full_state[i * 4: (i * 4) + 4])
            t_cov = deepcopy(self.full_cov[i * 4: (i * 4) + 4,
                             i * 4: (i * 4) + 4])
            u.set_state_cov(t_state, t_cov)
        self.targets = after_consensus_targets

    def update_trackers(self, i, pre_consensus=True):
        if pre_consensus:
            self.observations[i] = self.measurements
            self.node_positions[i] = self.position

            self.preconsensus_positions[i] = [t.state for t in self.targets]
            self.preconsensus_target_covs[i] = [t.state_cov for t in self.targets]
        else:
            self.consensus_positions[i] = [t.state for t in self.targets]
            self.consensus_target_covs[i] = [t.state_cov for t in self.targets]







