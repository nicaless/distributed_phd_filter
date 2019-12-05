import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.vq as vq
from scipy.stats import norm
from target import Target


class PHDFilter:
    def __init__(self,
                 J=10,
                 region=[(-100, 100), (-100, 100)]
                 ):
        self.target_model = Target()
        self.survival_prob = 0.9
        self.J = J
        self.region = region
        self.detection_probability = .98
        self.clutter_prob = 0

        self.particles = {}
        self.current_particles = []
        self.num_current_particles = 0
        self.weights = {}
        self.current_weights = []

        # prediction results
        self.predicted_pos = []
        self.predicted_weights = []
        self.predicted_num_targets = len(self.predicted_pos)

        # update results
        self.updated_weights = []

        # resample results
        self.resampled_pos = []
        self.resampled_weights = []
        self.resampled_num_targets = len(self.resampled_pos)

        # target centroids
        self.centroids = []
        self.centroid_movement = {}

    def predict(self):
        # Get New Positions for Existing Particles
        # Update Weights for Targets using Survival Probability
        # TODO: static survival probability, should be a function of position relative to FoV (region)
        new_positions = []
        new_weights = []
        for i, p in enumerate(self.current_particles):
            w = self.current_weights[i] * self.survival_prob
            p = self.target_model.next_state(x=p)
            new_positions.append(p)
            new_weights.append(w)

        # Sample from Birth Model
        # Assign weights to new births
        birth_particles = self.birth(self.J)
        birth_weights = [1. / self.J for i in range(self.J)]

        new_positions = new_positions + birth_particles
        new_weights = new_weights + birth_weights

        self.predicted_pos = new_positions
        self.predicted_weights = new_weights
        self.predicted_num_targets = len(self.predicted_pos)

    def birth(self, N):
        new_particles = []
        for k in range(0, N):
            # uniform birth
            x = np.random.uniform(low=self.region[0][0],
                                  high=self.region[0][1])
            y = np.random.uniform(low=self.region[1][0],
                                  high=self.region[1][1])
            new_particles.append(np.array([[x], [y], [0.], [0.]]))

        # return newborn targets and positions
        return new_particles

    # for each measurement/particle combo calculate:
    #   psi = detection_probability * probability we receive measurement at that particle position
    #                                  (should be a func of distance between measure and particle, difference should be less than measurement variance)
    # for each measurement calculate:
    #   Ck = sum(psi * weight of particle)
    # for each particle calculate:
    #   1 - detection probability (detection_probability should be almost 1 if in FoV, 0 otherwise)
    # + sum ( psi / clutter probability for measruement + Ck
    # multiple above by old weight
    def update(self, measurements):
        psi_mat = self.CalcPsi(measurements, self.predicted_pos)
        Ck_m = self.CalcCk(psi_mat)

        new_weights = []
        for i, p in enumerate(self.predicted_pos):
            sum_psi = 0
            Ck = sum(Ck_m)
            for m, m_psi in psi_mat.items():
                sum_psi += m_psi[i]
            w = (1 - self.DetectionProb(p)) + (sum_psi / (self.clutter_prob + Ck))
            new_weights.append(w * self.predicted_weights[i])
        self.updated_weights = new_weights

    def resample(self):
        particle_mass = np.sum(self.updated_weights)
        true_target_indices = Resample(np.array(self.updated_weights) / particle_mass)
        print(true_target_indices)
        true_particles = [self.predicted_pos[i] for i in true_target_indices]
        true_weights = [self.updated_weights[i] * np.ceil(particle_mass)
                        for i in true_target_indices]

        self.resampled_pos = true_particles
        self.resampled_weights = true_weights
        self.resampled_num_targets = len(self.resampled_pos)

    def estimate(self):
        particle_positions_matrix = np.zeros((len(self.resampled_pos), 2))
        for p in range(len(self.resampled_pos)):
            particle_positions_matrix[p][0] = self.resampled_pos[p][0]
            particle_positions_matrix[p][1] = self.resampled_pos[p][1]
        estimated_total_targets = int(np.ceil(np.sum(self.resampled_weights)))

        # TODO: fix...
        # estimated_total_targets = max(estimated_total_targets,
        #                               len(self.birth_models))
        if len(self.resampled_pos) == 1:
            self.centroids = particle_positions_matrix
        else:
            centroids, idx = vq.kmeans2(x, estimated_total_targets)
            self.centroids = centroids

    # psi = detection_prob * prob we receive measurement particle pos
    def CalcPsi(self, measurements, positions, measurement_variance=1):
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
        plt.scatter(x, y, label='centroid', color='black')

        plt.legend()
        plt.savefig('{folder}/{k}.png'.format(folder=folder, k=k))
        plt.clf()

    # TODO: add a reset
    def step_through(self, measurements, folder='results'):
        for i, m in measurements.items():
            print(i)
            self.predict()
            self.update(m)
            self.resample()
            self.estimate()
            self.plot(i, folder=folder)

            self.targets[i] = self.resampled_pos
            self.current_targets = self.resampled_pos
            self.num_current_targets = len(self.current_targets)
            self.weights[i] = self.resampled_weights
            self.current_weights = self.resampled_weights
            self.centroid_movement[i] = self.centroids

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
            plt.plot(x, y, label=t)

        plt.legend()
        plt.savefig('centroids.png')


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








