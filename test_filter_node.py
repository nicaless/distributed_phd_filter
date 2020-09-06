from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


region = [(-50, 50), (-50, 50)]

corner0 = Target(init_state=np.array([[region[0][0] + 10],
                                      [region[1][0] + 10],
                                      [0.1], [0.1]]))
corner1 = Target(init_state=np.array([[region[0][0] + 10],
                                      [region[1][1] - 10],
                                      [0.1], [0.1]]), dt_2=-1)
corner2 = Target(init_state=np.array([[region[0][1] - 10],
                                      [region[1][1] - 10],
                                      [0.1], [0.1]]), dt_1=-1, dt_2=-1)
corner3 = Target(init_state=np.array([[region[0][1] - 10],
                                      [region[1][0] + 10],
                                      [0.1], [0.1]]), dt_1=-1)
birthgmm = [corner0, corner1, corner2, corner3]


class FilterNodeTests(TestCase):

    def setUp(self):
        super().setUp()

        np.random.seed(42)
        self.N = 30
        self.generator = SimGenerator()
        self.generator.generate(steps=self.N)
        self.generator.plot()

        self.node_full = PHDFilterNode(0, birthgmm)
        self.node_sub = PHDFilterNode(0, birthgmm, region=[(-25, 25),
                                                           (-25, 25)])

    def tearDown(self):
        super().tearDown()

    # def test_predict(self):
    #     print('predict')
    #     self.node_full.predict()
    #
    #     assert len(self.node_full.predicted_pos) == 4
    #     assert len(self.node_full.predicted_targets) == 4
    #     assert self.node_full.predicted_pos[0].shape == (2, 1)
    #     assert self.node_full.predicted_targets[0].state.shape == (4, 1)
    #
    # def test_predict_sub(self):
    #     print('predict sub')
    #     self.node_sub.predict()
    #
    #     assert len(self.node_sub.predicted_pos) == 4
    #     assert len(self.node_sub.predicted_targets) == 4
    #     assert self.node_sub.predicted_pos[0].shape == (2, 1)
    #     assert self.node_sub.predicted_targets[0].state.shape == (4, 1)
    #
    # def test_update(self):
    #     print('update')
    #     self.node_full.predict()
    #
    #     m = self.generator.observations[0]
    #     self.node_full.update(m)
    #
    #     assert len(self.node_full.updated_targets) >= 4
    #
    # def test_update_sub(self):
    #     print('update sub')
    #     self.node_sub.predict()
    #     predicted_states = [t.state for t in self.node_sub.predicted_targets]
    #
    #     m = self.generator.observations[0]
    #     self.node_sub.update(m)
    #     updated_states = [t.state for t in self.node_sub.updated_targets]
    #
    #     assert len(self.node_sub.updated_targets) == 4
    #
    #     assert_array_equal(predicted_states, updated_states)
    #
    # def test_prune(self):
    #     print('prune')
    #     self.node_full.predict()
    #     m = self.generator.observations[0]
    #     self.node_full.update(m)
    #
    #     thresh = self.node_full.prune_thresh
    #     updated_targets = self.node_full.updated_targets
    #
    #     num_greater_than_thresh = len(list(
    #         filter(lambda comp: comp.weight > thresh, updated_targets)))
    #
    #     self.node_full.prune()
    #     assert num_greater_than_thresh == len(self.node_full.pruned_targets)
    #
    # def test_prune_sub(self):
    #     print('prune sub')
    #     self.node_sub.predict()
    #     m = self.generator.observations[13]
    #     self.node_sub.update(m)
    #
    #     thresh = self.node_sub.prune_thresh
    #     updated_targets = self.node_sub.updated_targets
    #
    #     print([comp.weight for comp in self.node_sub.updated_targets])
    #
    #     num_greater_than_thresh = len(list(
    #         filter(lambda comp: comp.weight > thresh, updated_targets)))
    #
    #     self.node_sub.prune()
    #     print(len(self.node_sub.updated_targets),
    #           len(self.node_sub.pruned_targets))
    #     assert num_greater_than_thresh == len(self.node_sub.pruned_targets)
    #
    # def test_merge(self):
    #     print('merge')
    #     self.node_full.predict()
    #     m = self.generator.observations[0]
    #     print(m)
    #     self.node_full.update(m)
    #     updated_weights = [comp.weight for comp in self.node_full.updated_targets]
    #
    #     self.node_full.prune()
    #     pruned_weights = [comp.weight for comp in self.node_full.pruned_targets]
    #
    #     self.node_full.merge()
    #     merged_weights = [comp.weight for comp in self.node_full.merged_targets]
    #
    #     print(updated_weights)
    #     print(pruned_weights)
    #     print(merged_weights)
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   pruned_weights, merged_weights)
    #
    # def test_merge_sub(self):
    #     print('merge sub')
    #     self.node_sub.predict()
    #     m = self.generator.observations[0]
    #     print(m)
    #     self.node_sub.update(m)
    #     updated_weights = [comp.weight for comp in
    #                        self.node_sub.updated_targets]
    #
    #     self.node_sub.prune()
    #     pruned_weights = [comp.weight for comp in
    #                       self.node_sub.pruned_targets]
    #
    #     self.node_sub.merge()
    #     merged_weights = [comp.weight for comp in
    #                       self.node_sub.merged_targets]
    #
    #     print(updated_weights)
    #     print(pruned_weights)
    #     print(merged_weights)
    #
    #     # assert_raises(AssertionError, assert_array_equal,
    #     #               pruned_weights, merged_weights)
    #
    #     assert_array_equal(pruned_weights, merged_weights)
    #
    # def test_reweight(self):
    #     print('reweight')
    #     self.node_full.predict()
    #     m = self.generator.observations[0]
    #     print(m)
    #     self.node_full.update(m)
    #     updated_weights = [comp.weight for comp in self.node_full.updated_targets]
    #
    #     self.node_full.prune()
    #     pruned_weights = [comp.weight for comp in self.node_full.pruned_targets]
    #
    #     self.node_full.merge()
    #     merged_weights = [comp.weight for comp in self.node_full.merged_targets]
    #
    #     self.node_full.reweight()
    #     reweighted_weights = [comp.weight for comp in self.node_full.reweighted_targets]
    #
    #     print(updated_weights)
    #     print(pruned_weights)
    #     print(merged_weights)
    #     print(reweighted_weights)
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   merged_weights, reweighted_weights)
    #
    # def test_reweight_sub(self):
    #     print('reweight sub')
    #     self.node_sub.predict()
    #     m = self.generator.observations[0]
    #     print(m)
    #     self.node_sub.update(m)
    #     updated_weights = [comp.weight for comp in
    #                        self.node_sub.updated_targets]
    #
    #     self.node_sub.prune()
    #     pruned_weights = [comp.weight for comp in
    #                       self.node_sub.pruned_targets]
    #
    #     self.node_sub.merge()
    #     merged_weights = [comp.weight for comp in
    #                       self.node_sub.merged_targets]
    #
    #     self.node_sub.reweight()
    #     reweighted_weights = [comp.weight for comp in
    #                           self.node_sub.reweighted_targets]
    #
    #     print(updated_weights)
    #     print(pruned_weights)
    #     print(merged_weights)
    #     print(reweighted_weights)
    #
    #     # assert_raises(AssertionError, assert_array_equal,
    #     #               merged_weights, reweighted_weights)
    #     assert_array_equal(merged_weights, reweighted_weights)

    # def test_sub_step_through(self):
    #     print('sub step through manual')
    #     self.node_sub.predict()
    #     m = self.generator.observations[0]
    #     self.node_sub.update(m)
    #     self.node_sub.prune()
    #     self.node_sub.merge()
    #     self.node_sub.reweight()
    #     self.node_sub.targets = self.node_sub.reweighted_targets
    #
    #     states_after_first_iter = [t.state for t in self.node_sub.targets]
    #     print('begin second iter')
    #     self.node_sub.predict()
    #
    #     states_new_predict = [t.state for t in self.node_sub.predicted_targets]
    #
    #     print(states_after_first_iter)
    #     print(len(states_after_first_iter))
    #     print(states_new_predict)
    #     print(len(states_new_predict))
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   states_after_first_iter, states_new_predict)
    #
    #     num_targets_after_second_predict = len(states_new_predict)
    #
    #     m = self.generator.observations[1]
    #     self.node_sub.update(m)
    #     self.node_sub.prune()
    #     num_targets_before_second_merge = len([t.state for t in
    #                                            self.node_sub.pruned_targets])
    #     self.node_sub.merge()
    #     num_targets_after_second_merge = len([t.state for t in
    #                                           self.node_sub.merged_targets])
    #
    #     assert num_targets_before_second_merge >= num_targets_after_second_merge
    #     print(num_targets_after_second_merge,
    #           num_targets_after_second_predict)
    #     assert num_targets_after_second_merge >= num_targets_after_second_predict

    # def test_step_through(self):
    #     print('step_through')
    #     self.node_full.step_through(self.generator.observations)
    #
    #     # Plot Target Positions Estimates and Truths
    #     for i, pos in self.node_full.preconsensus_positions.items():
    #         x = []
    #         y = []
    #         for p in pos:
    #             x.append(p[0])
    #             y.append(p[1])
    #         plt.scatter(x, y, label='estimates', alpha=0.5, s=20)
    #
    #         x = []
    #         y = []
    #         for p in self.generator.observations[i]:
    #             x.append(p[0])
    #             y.append(p[1])
    #         plt.scatter(x, y, label='truths', alpha=0.5, s=10)
    #
    #         plt.legend()
    #         plt.savefig('results/{i}.png'.format(i=i))
    #         plt.clf()

    def test_step_through_sub(self):
        print('step_through sub')
        self.node_sub.step_through(self.generator.observations)

        # for i in range(10, self.N):
        #     print(i)
        #     assert len(self.node_sub.preconsensus_positions[i]) > 4

        # Plot Target Positions Estimates and Truths
        for i, pos in self.node_sub.preconsensus_positions.items():
            ax = plt.axes()
            plt.xlim((-50, 50))
            plt.ylim((-50, 50))

            x = []
            y = []
            for p in pos:
                x.append(p[0])
                y.append(p[1])
            plt.scatter(x, y, label='estimates', alpha=0.5, s=20)

            x = []
            y = []
            for p in self.generator.observations[i]:
                x.append(p[0])
                y.append(p[1])
            plt.scatter(x, y, label='truths', alpha=0.5, s=10)

            # Plot FOV
            p = plt.Circle((self.node_sub.position[0],
                            self.node_sub.position[1]),
                           25, alpha=0.1)
            ax.add_patch(p)

            plt.legend()
            plt.savefig('results_node/{i}.png'.format(i=i))
            plt.clf()

