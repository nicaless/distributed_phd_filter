from copy import deepcopy
import math
import matplotlib.pyplot as plt
import numpy as np
from operator import attrgetter
import scipy.stats as ss

from target import Target


class PHDFilterNode:
    def __init__(self,
                 node_id,
                 birthgmm,
                 prune_thresh=1e-6,
                 # merge_thresh=0.01,
                 merge_thresh=1,
                 max_comp=100,
                 region=[(-50, 50), (-50, 50)]
                 ):
        self.node_id = node_id
        self.birthgmm = birthgmm
        self.prune_thresh = prune_thresh
        self.merge_thresh = merge_thresh
        self.max_components = max_comp
        self.region = region
        self.targets = []
        self.survival_prob = 0.98
        self.detection_probability = 0.95
        self.clutter_intensity = 0.0005  # clutter total is 5
        # TODO parametrize based on region and clutter intensity total (clutter lambda)

        # prediction results
        self.predicted_pos = []
        self.predicted_targets = []

        # update results
        self.updated_targets = []

        # pruned results
        self.pruned_targets = []

        # merged results
        self.merged_targets = []

        # fusion results
        self.fusion_targets = []

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
        for p in all_targets:
            predicted_pos.append(p.get_measurement())

        self.predicted_pos = predicted_pos
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
                                            dt_2=comp.dt_2))
                weightsum += newcomp_weight

            # Scale Weights
            reweighter = 1.0 / (self.clutter_intensity + weightsum)
            for comp in newgmmpartial:
                comp.weight *= reweighter

            newgmm.extend(newgmmpartial)

        self.updated_targets = newgmm

    def check_measure_oob(self, m):
        x = m[0][0]
        y = m[1][0]

        x_out_of_bounds = x < self.region[0][0] or x > self.region[0][1]
        y_out_of_bounds = y < self.region[1][0] or y > self.region[1][1]

        return x_out_of_bounds or y_out_of_bounds

    def prune(self):
        prunedgmm = list(filter(lambda comp: comp.weight > self.prune_thresh,
                                self.updated_targets))
        self.pruned_targets = prunedgmm

    def merge(self):
        weightsums = sum([comp.weight for comp in self.pruned_targets])
        # if weightsums == 0:
        #
        sourcegmm = [deepcopy(comp) for comp in self.pruned_targets]
        newgmm = []

        while len(sourcegmm) > 0:

            # Get Weightiest Component delete
            w = np.argmax(comp.weight for comp in sourcegmm)
            weightiest = sourcegmm[w]
            del sourcegmm[w]

            # Find all nearby components and delete
            # distances = [float(
            #     np.dot(
            #         np.dot((comp.state - weightiest.state).T,
            #                np.linalg.inv(comp.state_cov)),
            #         comp.state - weightiest.state))
            #     for comp in sourcegmm]
            distances = [math.hypot(comp.state[0][0] - weightiest.state[0][0],
                                    comp.state[1][0] - weightiest.state[1][0]
                                    ) for comp in sourcegmm]

            tosubsume = np.array([d <= self.merge_thresh for d in distances])
            subsumed = [weightiest]
            if any(tosubsume):
                subsumed.extend(list(np.array(sourcegmm)[tosubsume]))
                sourcegmm = list(np.array(sourcegmm)[~tosubsume])

            # Create new component from subsumed components
            newcomp_weight = sum([comp.weight for comp in subsumed])
            # newcomp_dt_1 = ss.mode([comp.dt_1 for comp in subsumed])[0][0]
            # newcomp_dt_2 = ss.mode([comp.dt_2 for comp in subsumed])[0][0]

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
        newweightsum = sum([comp.weight for comp in newgmm])
        self.merged_targets = newgmm[:self.max_components]
        weightnorm = newweightsum / float(weightsums)
        for comp in newgmm:
            comp.weight *= weightnorm
        self.merged_targets = newgmm

    def plot(self, k, folder='results', cardinality=None):
        # Plot Extracted States

        plt.xlim(self.region[0])
        plt.ylim(self.region[1])

        x, y = self.extractstates(cardinality=cardinality)
        plt.scatter(x, y)

        # plt.legend()
        plt.savefig('{folder}/{k}.png'.format(folder=folder, k=k))
        plt.clf()

    # TODO: add a reset
    def step_through(self, measurements, measurement_id=0, folder='results'):
        if not isinstance(measurements, dict):
            self.predict()
            self.update(measurements)
            self.prune()
            self.merge()
            self.targets = self.merged_targets
            self.plot(measurement_id, folder=folder)

        else:
            for i, m in measurements.items():
                self.predict()
                self.update(m)
                self.prune()
                self.merge()
                self.targets = self.merged_targets
                self.plot(i, folder=folder)

    def extractstates(self, cardinality=None, thresh=0.5):
        x = []
        y = []
        if cardinality is not None:
            all_targets = self.targets[:cardinality]
        else:
            all_targets = self.targets

        for comp in all_targets:
            if cardinality is None:
                if comp.weight > thresh:
                    for _ in range(int(np.ceil(comp.weight))):
                        x.append(comp.state[0][0])
                        y.append(comp.state[1][0])
            else:
                x.append(comp.state[0][0])
                y.append(comp.state[0][0])
        return x, y


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








