from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq as vq
from scipy.stats import norm
from target import Target


class PHDFilterNode:
    def __init__(self,
                 birthgmm,
                 region=[(-50, 50), (-50, 50)]
                 ):
        self.birthgmm = birthgmm
        self.targets = []
        self.survival_prob = 0.98
        self.detection_probability = 0.95
        self.clutter_intensity = 0.005  # clutter total is 5
        self.region = region

        # prediction results
        self.predicted_pos = []
        self.predicted_weights = []
        self.predicted_targets = []

        # update results
        self.updated_weights = []

        # # resample results
        # self.resampled_pos = []
        # self.resampled_weights = []
        # self.resampled_num_targets = len(self.resampled_pos)
        #
        # # target centroids
        # self.centroids = []
        # self.centroid_movement = {}
        # self.est_num_targets = len(self.centroids)

        # rescaled weights (after fusion)
        self.rescaled_weights = []
        self.adjusted_centroids = []

    def predict(self):
        # Existing Targets
        updated = [deepcopy(t) for t in self.targets]
        for t in updated:
            t.next_state()
            t.weight = t.weight * self.survival_prob

        # new born targets
        born = [deepcopy(t) for t in self.birthgmm]

        all_targets = updated + born
        predicted_pos = []
        predicted_weights = []
        for p in all_targets:
            predicted_pos.append(p.get_measurement())
            predicted_weights.append(p.weight)
        self.predicted_pos = predicted_pos
        self.predicted_weights = predicted_weights
        self.predicted_targets = all_targets

    def update(self, measurements):
        # Create Update Components
        nu = self.predicted_pos  # expected observation from prediction
        s = [t.measure_cov for t in self.predicted_targets]  # cov of expected observation from prediction

        K = [np.dot(np.dot(comp.state_cov, comp.H.T),
                    np.linalg.inv(s[index]))
             for index, comp in enumerate(self.predicted_targets)]
        PKK = [np.dot(np.eye(len(K[index])) - np.dot(K[index], comp.H),
                      comp.state_cov)
               for index, comp in enumerate(self.predicted_targets)]

        newgmm = [Target(init_weight=comp.weight * (1.0 - self.detection_probability),
                         init_state=comp.state,
                         init_cov=comp.state_cov)
                  for comp in self.predicted_targets]

        for m in measurements:
            newgmmpartial = []
            weightsum = 0
            for index, comp in enumerate(self.predicted_targets):
                obs_probability = dmvnorm(nu[index], s[index], m)
                newcomp_weight = comp.weight * self.detection_probability * obs_probability
                newcomp_state = comp.state + np.dot(K, m - nu[index])
                newcomp_state_cov = comp.state_cov
                newgmmpartial.append(Target(init_weight=newcomp_weight,
                                            init_state=newcomp_state,
                                            init_cov=newcomp_state_cov))
                weightsum += newcomp_weight

            # Scale Weights
            reweighter = 1.0 / (self.clutter_intensity + weightsum)
            for comp in newgmmpartial:
                comp.weight *= reweighter

            newgmm.extend(newgmmpartial)

        self.targets = newgmm
        new_weights = [t.weight for t in self.targets]
        self.updated_weights = new_weights

    def resample(self):
        particle_mass = np.sum(self.updated_weights)
        true_target_indices = Resample(np.array(self.updated_weights) / particle_mass)
        true_particles = [self.predicted_pos[i] for i in true_target_indices]
        true_weights = [self.updated_weights[i] * np.ceil(particle_mass)
                        for i in true_target_indices]

        self.resampled_pos = true_particles
        self.resampled_weights = true_weights
        self.resampled_num_targets = len(self.resampled_pos)

    def estimate(self, rescale=False):
        particle_positions_matrix = np.zeros((len(self.resampled_pos), 2))
        for p in range(len(self.resampled_pos)):
            particle_positions_matrix[p][0] = self.resampled_pos[p][0]
            particle_positions_matrix[p][1] = self.resampled_pos[p][1]
        if rescale:
            estimated_total_targets = int(np.round(np.sum(self.rescaled_weights), 0))
        else:
            estimated_total_targets = int(np.round(np.sum(self.resampled_weights), 0))

        # estimated_total_targets = max(estimated_total_targets,
        #                               len(self.birth_models))
        if estimated_total_targets == 0:
            centroids = []
        elif len(self.resampled_pos) == 1:
            centroids = particle_positions_matrix
        else:
            centroids, idx = vq.kmeans2(particle_positions_matrix,
                                        estimated_total_targets)
            centroids = centroids

        self.centroids = centroids

    # psi = detection_prob * prob we receive measurement particle pos
    def CalcPsi(self, measurements, positions, measurement_variance=20):
        psi_mat = {}
        for i, m in enumerate(measurements):
            psi_mat[i] = []
            x1 = m[0][0]
            y1 = m[1][0]
            for j, p in enumerate(positions):
                x2 = p[0][0]
                y2 = p[1][0]
                dist = abs(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
                prob_m_is_p = norm.sf(dist, 0, measurement_variance)
                psi = self.DetectionProb((x2, y2)) * prob_m_is_p
                psi_mat[i].append(psi)
        return psi_mat

    def DetectionProb(self, target):
        x = target[0]
        y = target[1]
        if x < self.region[0][0] or x > self.region[0][1] \
                or y < self.region[1][0] or y > self.region[1][1]:
            return 0
        else:
            return self.detection_probability

    # for each measurement calculate: Ck = sum(psi * weight of particle)
    def CalcCk(self, psi_mat):
        Ck_m = []
        for i, particles_psi in psi_mat.items():
            Ck = 0
            for j, psi in enumerate(particles_psi):
                Ck += psi * self.predicted_weights[j]
            Ck_m.append(Ck)
        return Ck_m

    def plot(self, k, folder='results'):
        # plot all 4 steps

        # plot predicted new positions of particles
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])

        x = []
        y = []
        for t in self.predicted_pos:
            x.append(t[0][0])
            y.append(t[1][0])
        plt.scatter(x, y, label='prediction')

        # plot resampled positions
        x = []
        y = []
        for t in self.resampled_pos:
            x.append(t[0][0])
            y.append(t[1][0])
        plt.scatter(x, y, label='resample')

        # plot centroid
        x = []
        y = []
        for t in self.centroids:
            x.append(t[0])
            y.append(t[1])
        plt.scatter(x, y, label='centroid', color='black', s=50)

        plt.legend()
        plt.savefig('{folder}/{k}.png'.format(folder=folder, k=k))
        plt.clf()

    # TODO: add a reset
    def step_through(self, measurements, folder='results'):
        for i, m in measurements.items():
            self.predict()
            self.update(m)
            self.resample()
            self.estimate()
            self.plot(i, folder=folder)

            self.particles[i] = self.resampled_pos
            self.current_particles = self.resampled_pos
            self.num_current_particles = len(self.current_particles)
            self.weights[i] = self.resampled_weights
            self.current_weights = self.resampled_weights
            self.centroid_movement[i] = self.centroids
            self.est_num_targets = len(self.centroids)

    def plot_centroids(self):
        targets = {}
        for i, cs in self.centroid_movement.items():
            for j, c in enumerate(cs):
                if j not in targets.keys():
                    targets[j] = [c]
                else:
                    targets[j].append(c)

        for t, pos in targets.items():
            x = []
            y = []
            for p in pos:
                x.append(p[0])
                y.append(p[1])
            plt.scatter(x, y, label=t)

        plt.legend()
        plt.savefig('centroids.png')

    def weightScale(self, avg_num_targets):
        rescaled_weights = []
        local_num_targets = len(self.centroids)
        if local_num_targets == 0:
            local_num_targets = 0.01
        for w in self.current_weights:
            scale = avg_num_targets / float(local_num_targets)
            rescaled_weights.append(scale * w)
        self.rescaled_weights = rescaled_weights


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


# Copied form here:
# https://filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html#systematic_resample
def Resample(weights):
    N = weights.size

    cum_sum_weights = np.cumsum(weights)

    random_positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')

    i, j = 0, 0
    while i < N:
        if random_positions[i] < cum_sum_weights[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    return np.unique(indexes)








