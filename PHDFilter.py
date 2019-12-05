import numpy as np
import matplotlib.pyplot as plt
from models import Resample
import math
from scipy.stats import norm


class PHDFilter:
    def __init__(self,
                 birth_models,
                 clutter_model,
                 measurement_model,
                 transition_model,
                 survival_model,
                 estimation_model,
                 init_targets=[],
                 init_weights=[],
                 J=10,
                 region=[(-100, 100), (-100, 100)]
                 ):
        # TODO: add region variable
        # TODO: update birth, clutter, measurement(?) models according to region(s) in view
        self.birth_models = birth_models
        self.clutter_model = clutter_model
        self.measurement_model = measurement_model
        self.transition_model = transition_model
        self.survival_model = survival_model
        self.estimation_model = estimation_model
        self.J = J
        self.region = region
        self.detection_probability = .98
        self.clutter_prob = 0

        # set of tracked positions of all targets
        # TODO: change naming to 'particles' instead of targets
        self.targets = {}
        self.current_targets = init_targets
        self.num_current_targets = len(self.current_targets)

        # set of weights of all targets
        self.weights = {}
        self.current_weights = init_weights

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
        # Get New Positions for Existing Targets
        # Update Weights for Targets using Survival Model
        new_positions = []
        new_weights = []
        for i, t in enumerate(self.current_targets):
            w = self.current_weights[i]
            p = self.transition_model.AdvanceState(t)
            w = self.survival_model.Evolute(w)
            new_positions.append(p)
            new_weights.append(w)

        # Sample from Birth Model
        # Assign weights to new births
        for birth_model in self.birth_models:
            num_births, birth_pos = birth_model.Sample(max_N=self.J)
            birth_weights = birth_model.Weight(num_births)

            # Merge
            new_positions = new_positions + birth_pos
            new_weights = new_weights + birth_weights

        self.predicted_pos = new_positions
        self.predicted_weights = new_weights
        self.predicted_num_targets = len(self.predicted_pos)

    # TODO: fix. resulting weights seem incorrect, need to add way more weight to measurement
    # for each measurement/particle combo calculate:
    #   psi = detection_probability * probability we receive measurement at that particle position
    #                                  (should be a func of distance between measure and particle, difference should be less than measurement variance)
    # for each measurement calculate:
    #   Ck = sum(psi * weight of particle)
    # for each particle calculate:
    #   1 - detection probability (detection_probability should be almost 1 if in FoV, 0 otherwise)
    # + sum ( psi / clutter probability for measruement + Ck
    # multiple above by old weight
    def update2(self, measurements):
        psi_mat = self.CalcPsi(measurements, self.predicted_pos)
        Ck_m = self.CalcCk(psi_mat)

        new_weights = []
        for i, p in enumerate(self.predicted_pos):
            sum_psi = 0
            Ck = sum(Ck_m)
            for m, m_psi in psi_mat.items():
                sum_psi += m_psi[i]
            w = (1 - self.DetectionProb(p)) + (sum_psi / Ck)
            new_weights.append(w * self.predicted_weights[i])
        self.updated_weights = new_weights

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

    def update(self, measurements):
        # Get the Weight Update
        weight_update = np.zeros(np.array(self.predicted_weights).shape)
        for m in measurements:
            likelihoods = np.array(
                [self.measurement_model.CalcWeight(m, t)
                 for t in self.predicted_pos])
            clutter_likelihoods = np.array(
                [self.clutter_model.Likelihood()
                 for t in self.predicted_pos])
            v = likelihoods / (clutter_likelihoods +
                               likelihoods *
                               np.array(self.predicted_weights))
            weight_update += v

        # Get Detection Probability
        un_detection_probs = []
        for i, t in enumerate(self.predicted_pos):
            detection_prob = self.measurement_model.DetectionProbability(t)
            un_detection_probs.append(1-detection_prob)

        # Update Weight
        new_weights = np.array(self.predicted_weights) * \
                      np.array(un_detection_probs) * \
                      weight_update

        self.updated_weights = new_weights

    def resample(self):
        particle_mass = np.sum(self.updated_weights)
        # print(particle_mass)
        # print(self.updated_weights)
        x = np.array(self.updated_weights) / particle_mass
        # print(x)
        # print(np.sum(x))
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
        estimated_total_targets = max(estimated_total_targets,
                                      len(self.birth_models))
        if len(self.resampled_pos) == 1:
            self.centroids = particle_positions_matrix
        else:
            centroids = self.estimation_model.estimate(particle_positions_matrix,
                                                       estimated_total_targets)
            self.centroids = centroids

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









