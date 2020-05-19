from copy import deepcopy
import math
from numba import jit, njit, cuda
import numpy as np
from operator import attrgetter
from scipy.spatial.distance import mahalanobis
from scipy.linalg import orth

from target import Target, DEFAULT_H


class DKFNode:
    def __init__(self,
                 node_id,
                 target,
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

        self.targets = [target]

        # prediction results
        self.predicted_pos = []
        self.predicted_targets = []

        # update results
        self.measurements = []
        self.updated_targets = []

        # TRACKERS
        self.observations = {}
        self.node_positions = {}

        self.preconsensus_positions = {}
        self.preconsensus_target_covs = {}

        self.consensus_positions = {}
        self.consensus_target_covs = {}

    def update_position(self, new_position):
        self.position = new_position
        x = new_position[0]
        y = new_position[1]
        self.region = [(x - self.fov, x + self.fov),
                       (y - self.fov, y + self.fov)]

    # TODO: Revise so thate states/cov is block diag of all targets
    def predict(self, input=None):
        # Existing Targets
        all_targets = self.targets

        # Set R for All Targets then get measurement
        predicted_pos = []
        for p in all_targets:
            p.next_state(input=input)

            p.R = self.R
            new_meas = p.get_measurement()
            predicted_pos.append(new_meas)

        self.predicted_pos = predicted_pos
        self.predicted_targets = all_targets

    # TODO: Revise so thate states/cov is block diag of all targets
    def update(self, measurements):
        updated_targets = []

        # TODO: use for loop to to create appropriate block diags
        for i, m in enumerate(measurements):
            if self.check_measure_oob(m) or \
                            np.random.rand() > self.detection_probability:
                # TODO: don't include in blockdiag
                measurement = self.measurements[i]
                R = np.eye(self.R.shape[0])
            else:
                measurement = m
                R = self.R

            H = self.predicted_targets[i].H
            cov = self.predicted_targets[i].state_cov
            predicted_state = self.predicted_targets[i].state

            G = self.predicted_targets[i].B
            T = np.concatenate((G, orth(G)), axis=1)
            T = np.linalg.inv(T + np.ones(T.shape) * 0.0001)
            x = np.concatenate((np.zeros((2, 2)), np.eye(2)), axis=1)
            L = np.dot(x, T)

            new_cov = self.gamma * np.dot(H.T, np.dot(np.linalg.inv(R), H)) + \
                      np.dot(L.T, np.dot(np.linalg.inv(np.dot(L, np.dot(cov, L.T))), L))

            IM = np.dot(H, predicted_state)
            K = np.dot(cov, np.dot(H.T, np.linalg.inv(R)))

            new_state = predicted_state + self.gamma * np.dot(K, (measurement - IM))

            updated_targets.append(Target(init_state=new_state,
                                          init_cov=new_cov))

        self.measurements = measurements
        self.updated_targets = updated_targets

    def check_measure_oob(self, m):
        if m is None:
            return True

        x = m[0][0]
        y = m[1][0]

        x_out_of_bounds = x < self.region[0][0] or x > self.region[0][1]
        y_out_of_bounds = y < self.region[1][0] or y > self.region[1][1]

        return x_out_of_bounds or y_out_of_bounds

    def step_through(self, measurements, measurement_id=0):
        if not isinstance(measurements, dict):
            self.predict()
            self.update(measurements)
            self.targets = self.updated_targets
            self.update_trackers(measurement_id)
        else:
            for i, m in measurements.items():
                self.predict()
                self.update(m)
                self.targets = self.updated_targets
                self.update_trackers(i)

    def update_trackers(self, i, pre_consensus=True):
        if pre_consensus:
            self.observations[i] = self.measurements
            self.node_positions[i] = self.position

            self.preconsensus_positions[i] = [t.state for t in self.targets]
            self.preconsensus_target_covs[i] = [t.state_cov for t in self.targets]
        else:
            self.consensus_positions[i] = [t.state for t in self.targets]
            self.consensus_target_covs[i] = [t.state_cov for t in self.targets]

    @staticmethod
    def get_mahalanobis(target1, target2):
        d = mahalanobis(target1.state, target2.state,
                        np.linalg.inv(target1.state_cov))
        return d

@njit
def dmvnorm(state, cov, obs):
    """
    Evaluate a multivariate normal, given a state (vector) and covariance (matrix) and a position x (vector) at which to evaluate"
    :param state:
    :param cov:
    :param obs:
    :return:
    """
    k = state.shape[0]
    part1 = (2.0 * np.pi) ** (-k * 0.5)
    part2 = np.power(np.linalg.det(cov), -0.5)
    dist = obs - state
    part3 = np.exp(-0.5 * np.dot(np.dot(dist.T, np.linalg.inv(cov)), dist))
    return part1 * part2 * part3








