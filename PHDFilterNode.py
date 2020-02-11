from copy import deepcopy
import math
import numpy as np
from operator import attrgetter
from scipy.spatial.distance import mahalanobis

from target import Target


class PHDFilterNode:
    def __init__(self,
                 node_id,
                 birthgmm,
                 prune_thresh=1e-6,
                 merge_thresh=0.2,
                 max_comp=100,
                 clutter_rate=5,
                 # clutter_rate=0,
                 position=np.array([0, 0, 0]),
                 region=[(-50, 50), (-50, 50)]
                 ):
        self.node_id = node_id

        self.position = position
        self.region = region
        self.R = np.eye(2)
        self.fov = (region[0][1] - region[0][0]) / 2.0
        area = (region[0][1] - region[0][0]) * (region[1][1] - region[1][0])
        self.clutter_intensity = clutter_rate / area

        self.prune_thresh = prune_thresh
        self.merge_thresh = merge_thresh
        self.max_components = max_comp

        self.survival_prob = 0.98
        self.detection_probability = 0.95

        self.birthgmm = birthgmm
        self.targets = []

        # prediction results
        self.predicted_pos = []
        self.predicted_targets = []

        # update results
        self.measurements = []
        self.updated_targets = []

        # pruned results
        self.pruned_targets = []

        # merged results
        self.merged_targets = []

        # reweighted results
        self.reweighted_targets = []

        # TRACKERS
        self.observations = {}
        self.node_positions = {}
        self.detection_probs = {}

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

    def predict(self):
        # Existing Targets
        updated = [deepcopy(t) for t in self.targets]
        for t in updated:
            t.next_state()
            t.weight = t.weight * self.survival_prob
        keep_updated = [t for t in updated]

        # new born targets
        born = [deepcopy(t) for t in self.birthgmm]
        for b in born:
            b.weight = b.weight / float(len(born))

        all_targets = keep_updated + born

        # Set R for All Targets then get measurement
        predicted_pos = []
        for p in all_targets:
            p.R = self.R
            new_meas = p.get_measurement()
            predicted_pos.append(new_meas)

        self.predicted_pos = predicted_pos
        self.predicted_targets = all_targets

    def add_clutter(self):
        pass

    def update(self, measurements):
        self.measurements = measurements

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
                         init_cov=comp.state_cov,
                         dt_1=comp.dt_1,
                         dt_2=comp.dt_2)
                  for comp in self.predicted_targets]

        for m in measurements:
            if self.check_measure_oob(m) or \
                            np.random.rand() > self.detection_probability:
                continue
            newgmmpartial = []
            weightsum = 0
            for index, comp in enumerate(self.predicted_targets):
                obs_probability = dmvnorm(nu[index], s[index], m)
                newcomp_weight = float(comp.weight *
                                       self.detection_probability *
                                       obs_probability)
                newcomp_state = comp.state + np.dot(K[index], m - nu[index])
                newcomp_state_cov = comp.state_cov
                newgmmpartial.append(Target(init_weight=newcomp_weight,
                                            init_state=newcomp_state,
                                            init_cov=newcomp_state_cov,
                                            dt_1=comp.dt_1,
                                            dt_2=comp.dt_2
                                            ))
                weightsum += newcomp_weight

            # Scale Weights
            reweighter = 1.0 / (self.clutter_intensity + weightsum)
            for comp in newgmmpartial:
                comp.weight *= reweighter

            newgmm.extend(newgmmpartial)

        self.updated_targets = newgmm

    def check_measure_oob(self, m):
        # # TODO: investigate why there have nan states/measurements anyway
        # any_nan = np.isnan(m).any()

        x = m[0][0]
        y = m[1][0]

        x_out_of_bounds = x < self.region[0][0] or x > self.region[0][1]
        y_out_of_bounds = y < self.region[1][0] or y > self.region[1][1]

        return x_out_of_bounds or y_out_of_bounds

    @staticmethod
    def get_mahalanobis(target1, target2):
        d = mahalanobis(target1.state, target2.state,
                        np.linalg.inv(target1.state_cov))
        return d

    def prune(self):
        prunedgmm = list(filter(lambda comp: comp.weight > self.prune_thresh,
                                self.updated_targets))
        self.pruned_targets = prunedgmm

    def merge(self):
        sourcegmm = [deepcopy(comp) for comp in self.pruned_targets]
        newgmm = []

        while len(sourcegmm) > 0:

            # Get Weightiest Component delete
            w = np.argmax(comp.weight for comp in sourcegmm)
            weightiest = sourcegmm[w]
            del sourcegmm[w]

            # Find all nearby components to merge with
            distances = [self.get_mahalanobis(comp, weightiest)
                         for comp in sourcegmm]

            tosubsume = np.array([d <= self.merge_thresh for d in distances])
            subsumed = [weightiest]
            if any(tosubsume):
                subsumed.extend(list(np.array(sourcegmm)[tosubsume]))
                sourcegmm = list(np.array(sourcegmm)[~tosubsume])

            # Create new component from subsumed components
            newcomp_weight = sum([comp.weight for comp in subsumed])

            newcomp_state = np.sum(
                np.array([
                    comp.weight * comp.state
                    for comp in subsumed]), 0) / newcomp_weight

            newcomp_cov = np.sum(
                np.array([
                    comp.weight *
                    (comp.state_cov + (weightiest.state - comp.state) *
                     (weightiest.state - comp.state).T)
                    for comp in subsumed]), 0) / newcomp_weight

            newcomp = Target(init_weight=newcomp_weight,
                             init_state=newcomp_state,
                             init_cov=newcomp_cov,
                             dt_1=weightiest.dt_1,
                             dt_2=weightiest.dt_2)

            newgmm.append(newcomp)

        # Keep no more than max components
        newgmm.sort(key=attrgetter('weight'), reverse=True)
        self.merged_targets = newgmm[:self.max_components]

    def reweight(self):
        weightsums = sum([comp.weight for comp in self.updated_targets])
        newweightsum = sum([comp.weight for comp in self.merged_targets])
        self.reweighted_targets = [deepcopy(comp)
                                   for comp in self.merged_targets]
        if newweightsum > 0:
            weightnorm = float(weightsums) / newweightsum
            for comp in self.reweighted_targets:
                comp.weight *= weightnorm

    def step_through(self, measurements, measurement_id=0):
        if not isinstance(measurements, dict):
            self.predict()
            self.update(measurements)
            self.prune()
            self.merge()
            self.reweight()
            self.targets = self.reweighted_targets
            self.update_trackers(measurement_id)
        else:
            for i, m in measurements.items():
                self.predict()
                self.update(m)
                self.prune()
                self.merge()
                self.reweight()
                self.targets = self.reweighted_targets
                self.update_trackers(i)

    def update_trackers(self, i, pre_consensus=True):
        if pre_consensus:
            self.observations[i] = self.measurements
            self.node_positions[i] = self.position
            self.detection_probs[i] = self.detection_probability

            self.preconsensus_positions[i] = [np.array([[t.state[0][0]],
                                                        [t.state[1][0]]])
                                              for t in self.targets]
            self.preconsensus_target_covs[i] = [t.state_cov
                                                for t in self.targets]
        else:
            self.consensus_positions[i] = [np.array([[t.state[0][0]],
                                                     [t.state[1][0]]])
                                           for t in self.targets]
            self.consensus_target_covs[i] = [t.state_cov for t in self.targets]


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








