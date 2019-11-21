"""

Many models taken from https://github.com/hichamjanati/PHD-filter
TODO: change to update using this https://github.com/danstowell/gmphd/blob/master/gmphd.py ???

"""

import numpy as np
from scipy.stats import mvn


def pseudo_det(X):
    eig = np.linalg.eig(X)[0]
    return np.prod(eig[eig > 0])


"""
Birth and Survival are models used for target generation and "un-generation"
"""


class Birth:
    def __init__(self, poisson_coef, poisson_mean, poisson_cov, poisson_window):
        self.poisson_coef = poisson_coef
        self.poisson_mean = poisson_mean
        self.poisson_cov = poisson_cov
        self.poisson_window = poisson_window

    """
    return number of newborn targets and positions
    """
    def birth(self):
        # cdf of multivariate gaussian over window [wmin;wmax]
        mu, i = mvn.mvnun(self.poisson_window[0], self.poisson_window[1],
                          self.poisson_mean, self.poisson_cov)
        mu *= self.poisson_coef

        # number of newborn targets
        N = np.random.poisson(mu)

        # generate position of N new targets within the window
        positions = []
        for k in range(0, N):
            x_k = np.random.multivariate_normal(mean=self.poisson_mean,
                                                cov=self.poisson_cov)
            while (min(x_k > self.poisson_window[0]) == False) or \
                    (min(self.poisson_window[1] > x_k) == False):
                x_k = np.random.multivariate_normal(mean=self.poisson_mean,
                                                    cov=self.poisson_cov)
            positions.append(x_k)

        # return number of newborn targets and positions
        return N, positions


class Survival:
    def __init__(self, survival_prob):
        self.survival_prob = survival_prob

    """
    return number of dead targets and their index
    """
    def survive(self, num_targets):
        survival = np.random.uniform(size=num_targets)
        deads = [i for i in num_targets if survival[i] >= self.survival_prob]
        return len(deads), deads


class TargetMotion:
    def __init__(self, X, cov):
        self.X = X
        self.cov = cov

    def next(self, xk_1):
        x = np.asarray(xk_1).reshape(-1)
        # Returns next true positions and velocities given previous ones (xk_1)
        return np.random.multivariate_normal(mean=np.dot(self.X, x),
                                             cov=self.cov)

    def next_pdf(self, xk_1, xk):
        xk = np.asarray(xk).reshape(-1)
        # Returns pdf of next true pos and velocities given xk_1) ?
        mean = np.dot(self.X, xk)

        sqrt_pseudo_det = pseudo_det(self.cov) ** 0.5
        inv_cov = np.linalg.pinv(self.cov)

        pdf = 1 / (2 * np.pi * sqrt_pseudo_det) * \
            np.exp(-0.5 * (xk_1 - mean).dot(inv_cov.dot((xk_1 - mean).T)))

        return pdf


# TODO: change this
class TargetQ:
    def __init__(self, X, cov):
        self.X = X
        self.cov = cov

    def next(self, xk_1):
        x = np.asarray(xk_1).reshape(-1)
        # Returns next true positions and velocities given previous ones (xk_1)
        return np.random.multivariate_normal(mean=np.dot(self.X, x),
                                             cov=self.cov)

    def next_pdf(self, xk_1, xk):
        xk = np.asarray(xk).reshape(-1)
        # Returns pdf of next true pos and velocities given xk_1) ?
        mean = np.dot(self.X, xk)

        sqrt_pseudo_det = pseudo_det(self.cov) ** 0.5
        inv_cov = np.linalg.pinv(self.cov)

        pdf = 1 / (2 * np.pi * sqrt_pseudo_det) * \
            np.exp(-0.5 * (xk_1 - mean).dot(inv_cov.dot((xk_1 - mean).T)))

        return pdf


class Measurement:
    def __init__(self, H, noise):
        self.H = H
        self.noise = noise

    def next(self, xk):
        xk = np.asarray(xk).reshape(-1)
        # Return observed position given true position x
        return np.random.multivariate_normal(mean=np.dot(self.H, xk),
                                             cov=self.noise)

    def next_pdf(self, yk, xk):
        mean = np.dot(self.H, xk)

        sqrt_det_g = np.linalg.det(self.noise) ** 0.5
        obs_inv = np.linalg.inv(self.noise)

        pdf = 1 / (2 * np.pi * sqrt_det_g) * \
              np.exp(-0.5*(yk - mean).dot(obs_inv.dot((yk - mean).T)))

        return pdf
