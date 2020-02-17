from unittest import TestCase

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from PHDFilterNetwork import PHDFilterNetwork
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
        self.N = 20
        self.generator = SimGenerator()
        self.generator.generate(steps=self.N)
        self.generator.plot()

        self.node_full = PHDFilterNode(0, birthgmm)

        self.node_attrs = {}
        num_nodes = 3
        x_start = -50 + (100.0 / (num_nodes + 1))
        pos_start = np.array([x_start, 0, 20])
        pos_init_dist = np.floor(100.0 / (num_nodes + 1))
        fov = 20
        for n in range(num_nodes):
            pos = pos_start + np.array([n * pos_init_dist, 0, 0])
            region = [(pos[0] - fov, pos[0] + fov),
                      (pos[1] - fov, pos[1] + fov)]
            self.node_attrs[n] = PHDFilterNode(n, birthgmm,
                                               position=pos,
                                               region=region)
        G = nx.Graph()
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)

        weight_attrs = {}
        for i in range(num_nodes):
            weight_attrs[i] = {}
            self_degree = G.degree(i)
            metropolis_weights = []
            for n in G.neighbors(i):
                degree = G.degree(n)
                mw = 1 / (1 + max(self_degree, degree))
                weight_attrs[i][n] = mw
                metropolis_weights.append(mw)
            weight_attrs[i][i] = 1 - sum(metropolis_weights)

        self.filternetwork = PHDFilterNetwork(self.node_attrs,
                                              weight_attrs, G)

    def tearDown(self):
        super().tearDown()

    # def test_get_k(self):
    #     pass
    #
    # def test_fuse_covs_geom(self):
    #     pass
    #
    # def test_fuse_states_geom(self):
    #     pass
    #
    # def test_fuse_alphas_geom(self):
    #     pass

    def test_fuse_geom(self):
        self.filternetwork.step_through(self.generator.observations[0],
                                        self.generator.true_positions[0],
                                        how='geom',
                                        opt='team',
                                        fail_int=[1],
                                        base=True,
                                        noise_mult=1)



    # def test_step_through_sub_manual(self):
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

    # def test_step_through_sub(self):
    #     print('step_through sub')
    #     self.node_sub.step_through(self.generator.observations)
    #
    #     # Plot Target Positions Estimates and Truths
    #     for i, pos in self.node_sub.preconsensus_positions.items():
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
