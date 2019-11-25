import numpy as np
from models import Resample


class PHDFilter:
    def __init__(self,
                 birth_model,
                 clutter_model,
                 measurement_model,
                 transition_model,
                 survival_model,
                 estimation_model,
                 init_targets=[],
                 init_weights=[],
                 J=10
                 ):
        self.birth_model = birth_model
        self.clutter_model = clutter_model
        self.measurement_model = measurement_model
        self.transition_model = transition_model
        self.survival_model = survival_model
        self.estimation_model = estimation_model
        self.J = J

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
        num_births, birth_pos = self.birth_model.Sample(max_N=self.J)
        birth_weights = self.birth_model.Weight(num_births)

        # Merge
        new_positions = new_positions + birth_pos
        new_weights = new_weights + birth_weights

        self.predicted_pos = new_positions
        self.predicted_weights = new_weights
        self.predicted_num_targets = len(self.predicted_pos)

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
        # print(self.updated_weights)
        # print(np.sum(self.updated_weights))

    def resample(self):
        particle_mass = np.sum(self.updated_weights)
        true_target_indices = Resample(np.array(self.updated_weights) / particle_mass)
        true_particles = [self.predicted_pos[i] for i in true_target_indices]
        true_weights = [self.updated_weights[i] for i in true_target_indices]

        self.resampled_pos = true_particles
        self.resampled_weights = true_weights
        self.resampled_num_targets = len(self.resampled_pos)

    def estimate(self):
        particle_positions_matrix = np.zeros((len(self.resampled_pos), 2))
        for p in range(len(self.resampled_pos)):
            particle_positions_matrix[p][0] = self.resampled_pos[p][0]
            particle_positions_matrix[p][1] = self.resampled_pos[p][1]
        estimated_total_targets = int(np.ceil(np.sum(self.resampled_weights)))
        centroids = self.estimation_model.estimate(particle_positions_matrix,
                                                   estimated_total_targets)
        self.centroids = centroids





