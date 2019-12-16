from unittest import TestCase

import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from PHDFilterNetwork import PHDFilterNetwork
from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


class GMPHDTest(TestCase):

    def setUp(self):
        super().setUp()

        self.target = Target()

        self.generator = SimGenerator(5, init_targets=[Target()])
        self.generator.generate(20)
        self.generator.plot(show_clutter=True)

        self.birthgmm_1 = []
        for x in range(-50, 0, 10):
            for y in range(-50, 50, 10):
                target = Target(init_weight=1,
                                init_state=np.array([[x], [y], [0.0], [0.0]]),
                                dt_1=0, dt_2=0)
                self.birthgmm_1.append(target)

        self.birthgmm_2 = []
        for x in range(-25, 25, 10):
            for y in range(-50, 50, 10):
                target = Target(init_weight=1,
                                init_state=np.array([[x], [y], [0.0], [0.0]]),
                                dt_1=0, dt_2=0)
                self.birthgmm_2.append(target)

        self.birthgmm_3 = []
        for x in range(0, 50, 10):
            for y in range(-50, 50, 10):
                target = Target(init_weight=1,
                                init_state=np.array(
                                    [[x], [y], [0.0], [0.0]]),
                                dt_1=0, dt_2=0)
                self.birthgmm_3.append(target)

        self.filternode_1 = PHDFilterNode(1, self.birthgmm_1,
                                          region=[(-50, 0), (-50, 50)])
        self.filternode_2 = PHDFilterNode(2, self.birthgmm_2,
                                          region=[(-25, 25), (-50, 50)])
        self.filternode_3 = PHDFilterNode(3, self.birthgmm_3,
                                          region=[(0, 50), (-50, 50)])

        self.G = nx.Graph()
        for i in range(1, 3):
            self.G.add_edge(i, i + 1)
        node_attrs = {1: self.filternode_1,
                      2: self.filternode_2,
                      3: self.filternode_3}
        weight_attrs = {1: 1.0/3, 2: 1.0/3, 3: 1.0/3}

        self.filternetwork = PHDFilterNetwork(node_attrs,
                                              weight_attrs,
                                              self.G)

    def tearDown(self):
        super().tearDown()

    # def test_network_setup(self):
    #     assert len(self.filternetwork.network.nodes) == 3
    #     assert nx.get_node_attributes(self.filternetwork.network,
    #                                   'node')[1] == self.filternode_1

    # def test_step_through_single(self):
    #     self.filternetwork.step_through(self.generator.observations[0])

    def test_step_through(self):
        self.filternetwork.step_through(self.generator.observations)

    # def test_step_through_arith(self):
    #     self.filternetwork.step_through(self.generator.observations,
    #                                     how='arith')

    # def test_share_info(self):
    #     self.filternetwork.step_through(self.generator.observations[0])
    #
    #     self.filternetwork.share_info(1)
    #     assert len(self.filternetwork.node_share) == 3
    #     assert len(self.filternetwork.node_share[1]['node_comps']) == \
    #            len(self.filternode_1.targets)
    #     assert len(self.filternetwork.node_share[1]['neighbor_comps'][2]) == \
    #            len(self.filternode_2.targets)

    # def test_get_closest_comps(self):
    #     self.filternetwork.step_through(self.generator.observations[0])
    #     # self.filternetwork.share_info(3)
    #     # self.filternetwork.get_closest_comps(3)
    #     print(self.filternetwork.node_keep)
    #
    #     assert len(self.filternetwork.node_keep[3]) == \
    #            len(self.filternode_3.targets)

    # def test_fuse_covs(self):
    #     self.filternetwork.step_through(self.generator.observations[0])
    #     self.filternetwork.share_info(1)
    #     self.filternetwork.get_closest_comps(1)
    #     new_covs = self.filternetwork.fuse_covs(1)
    #
    #     assert len(new_covs) == len(self.filternode_1.targets)

    # def test_fuse_states(self):
    #     self.filternetwork.step_through(self.generator.observations[0])
    #     self.filternetwork.share_info(1)
    #     self.filternetwork.get_closest_comps(1)
    #     new_states = self.filternetwork.fuse_states(1)
    #     print(self.filternode_1.targets[0].state)
    #
    #     assert len(new_states) == len(self.filternode_1.targets)

    # def test_fuse_alphas(self):
    #     self.filternetwork.step_through([self.generator.observations[0]])
    #     self.filternetwork.share_info(1)
    #     self.filternetwork.get_closest_comps(1)
    #     print(self.filternetwork.node_keep)
    #     new_covs = self.filternetwork.fuse_covs(1)
    #     new_states = self.filternetwork.fuse_states(1)
    #
    #     print(sum([t.weight for t in self.filternode_1.targets]))
    #     new_alphas = self.filternetwork.fuse_alphas(1, new_covs, new_states)
    #     print(new_alphas)
    #     print(sum(new_alphas))
    #     print(self.generator.observations[0])
    #
    #     assert len(new_alphas) == len(self.filternode_1.targets)

    # def test_update_comps(self):
    #     self.filternetwork.step_through(self.generator.observations[0])
    #     print(self.filternetwork.node_keep)
    #     # self.filternetwork.share_info(1)
    #     # self.filternetwork.get_closest_comps(1)
    #     # new_covs = self.filternetwork.fuse_covs(1)
    #     # new_states = self.filternetwork.fuse_states(1)
    #     #
    #     # new_alphas = self.filternetwork.fuse_alphas(1, new_covs, new_states)
    #
    #     assert len(self.filternode_1.predicted_targets) > len(self.filternode_1.targets)
    #     assert len(self.filternode_1.updated_targets) > len(self.filternode_1.targets)
    #     assert len(self.filternode_1.pruned_targets) > len(self.filternode_1.targets)
    #     print(len(self.filternode_1.merged_targets))
    #     print(len(self.filternode_1.targets))
    #     assert len(self.filternode_1.merged_targets) == len(self.filternode_1.targets)
