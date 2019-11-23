"""

Many models taken from https://github.com/rafaelkarrer/python-particle-phd-filter

"""

import numpy as np
from scipy.stats import multivariate_normal as mv_normal
from scipy.stats import uniform


class Birth:
    def __init__(self,
                 poisson_lambda,
                 region=[(-100, 100), (-100, 100)]):
        self.poisson_lambda = poisson_lambda
        self.region = region

    """
    returns position of newborn targets
    """
    def Sample(self):
        # number of newborn targets
        N = np.random.poisson(self.poisson_lambda)

        # generate position of N new targets within the region
        positions = []

        for k in range(0, N):
            x_center = (self.region[0][1] - self.region[0][0]) + \
                       self.region[0][0]
            y_center = (self.region[1][1] - self.region[1][0]) + \
                       self.region[1][0]
            x = np.random.poisson(x_center) + self.region[0][0]
            y = np.random.poisson(y_center) + self.region[1][0]
            positions.append(np.array([[x], [y], [0.], [0.]]))

        # return newborn targets and positions
        return N, positions


class Transition:
    def __init__(self, process_noise=0.1, step=1):
        self.A = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        # self.Q = var_v * np.array([[1.0 / 3, 0, 1.0 / 2, 0],
        #                             [0, 1.0 / 3, 0, 1.0 / 2],
        #                             [1.0 / 2, 0, 1, 0],
        #                             [0, 1.0 / 2, 0, 1]])
        # self.rv = mv_normal(np.array([0, 0, 0, 0]), self.Q)
        self.Q = np.eye(self.A.shape[0]) * process_noise ** 2
        self.B = np.eye(self.A.shape[0])
        self.U = np.zeros((self.A.shape[0], 1)) + step

    def AdvanceState(self, x):
        # v = np.array(self.rv.rvs(1)).T
        # return self.A * x + v
        next_state = np.dot(self.A, x) + np.dot(self.B, self.U)
        next_state[0, 0] = next_state[0, 0] + np.random.randn() * self.Q[0, 0]
        next_state[1, 0] = next_state[1, 0] + np.random.randn() * self.Q[1, 1]
        return next_state


class Survival:
    # Default probability is 0.9.
    # otherwise should depend on the region ?
    def __init__(self, survival_probability=0.9):
        self.survival_probability = survival_probability

    def Evolute(self, weights):
        # uniform survival, position of particles is not considered
        return weights * self.survival_probability


class Measurement:
    def __init__(self,
                 measurement_variance,
                 detection_probability,
                 region=[(-100, 100), (-100, 100)]):
        self.measurement_variance = measurement_variance
        self.detection_probability = detection_probability
        self.C = np.eye(len(region)) * self.measurement_variance
        self.likelihood_func = mv_normal(np.zeros(len(region)),
                                         self.C)
        self.detection_func = uniform()
        self.region = region

    def Sample(self, n):
        return self.likelihood_func.rvs(n)

    def Likelihood(self, measurement):
        return self.likelihood_func.pdf(measurement)

    def DetectionProbability(self, target):
        x = target[0]
        y = target[1]
        if x < self.region[0][0] or x > self.region[0][1] \
                or y < self.region[1][0] or y > self.region[1][1]:
            return 0
        else:
            return self.detection_probability

    def CalcWeight(self, measurement, target):
        return self.Likelihood(measurement) * self.DetectionProbability(target)

    def Measure(self, target):
        sample = self.Sample(target.shape[1] * 2).T
        n = sample.reshape(4, target.shape[1])
        z = target + n
        det = np.where(self.detection_func.rvs(target.shape[1]) <=
                       self.detection_probability)[0]
        z[2:4, :] = 0
        return z[:, det]


class Clutter:
    def __init__(self,
                 poisson_lambda,
                 region=[(-100, 100), (-100, 100)]):
        self.poisson_lambda = poisson_lambda
        self.region = region

    def Sample(self):
        if self.poisson_lambda == 0:
            return 0, []

        # amount of clutter
        N = np.random.poisson(self.poisson_lambda)

        # generate position of N clutter within the region
        positions = []

        for k in range(0, N):
            x_center = (self.region[0][1] - self.region[0][0]) + \
                       self.region[0][0]
            y_center = (self.region[1][1] - self.region[1][0]) + \
                       self.region[1][0]
            x = np.random.poisson(x_center) + self.region[0][0]
            y = np.random.poisson(y_center) + self.region[1][0]
            positions.append(np.array([x, y]))

        # return clutter count and positions
        return N, positions
