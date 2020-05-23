from unittest import TestCase

from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from DKFNetwork import DKFNetwork
from DKFNode import DKFNode
from target import Target


region = [(-50, 50), (-50, 50)]


class FilterNodeTests(TestCase):

    def setUp(self):
        super().setUp()

        np.random.seed(42)
        self.N = 20

        self.target1 = Target()
        self.target2 = Target()

        self.node0 = DKFNode(0, [deepcopy(self.target1),
                                 deepcopy(self.target2),
                                 ])

        self.node1 = DKFNode(1, [deepcopy(self.target1),
                                 deepcopy(self.target2),
                                 ])
        self.node_attrs = {0: self.node0, 1: self.node1}

        G = nx.Graph()
        G.add_edge(0, 1)

        weight_attrs = {}
        for i in range(2):
            weight_attrs[i] = {}
            self_degree = G.degree(i)
            metropolis_weights = []
            for n in G.neighbors(i):
                degree = G.degree(n)
                mw = 1 / (1 + max(self_degree, degree))
                weight_attrs[i][n] = mw
                metropolis_weights.append(mw)
            weight_attrs[i][i] = 1 - sum(metropolis_weights)
        self.network = DKFNetwork(self.node_attrs, weight_attrs, G,
                                  [self.target1, self.target2])

    def tearDown(self):
        super().tearDown()

    # def test_01_predict(self):
    #     print('predict')
    #
    #     node0_target_curr_pos = self.node0.targets[0].state
    #
    #     self.node0.predict()
    #
    #     node0_pred_full_state = self.node0.full_state_prediction
    #     node0_pred_full_cov = self.node0.full_cov_prediction
    #     node0_pred_pos = self.node0.predicted_pos[0]
    #     node0_target_new_pos = self.node0.predicted_targets[0].state
    #
    #     assert node0_pred_full_state is not None
    #     assert node0_pred_full_cov is not None
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_target_curr_pos, node0_target_new_pos)
    #
    #     assert_array_equal(node0_target_new_pos, node0_pred_pos)
    #
    # def test_02_update(self):
    #     print('update')
    #
    #     self.node0.predict()
    #     node0_pred_full_state = self.node0.full_state_prediction
    #     node0_pred_full_cov = self.node0.full_cov_prediction
    #     node0_target_new_pos = self.node0.predicted_targets[0].state
    #
    #     self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
    #                        np.array([[1.01], [1.01], [1.01], [1.01]])])
    #
    #     node0_update_full_state = self.node0.full_state_update
    #     node0_update_full_cov = self.node0.full_cov_update
    #     node0_target_update_pos = self.node0.updated_targets[0].state
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_target_new_pos, node0_target_update_pos)
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_pred_full_state, node0_update_full_state)
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_pred_full_cov, node0_update_full_cov)

    # def test_03_update_missing_measurement(self):
    #     print('update, missing measurement')
    #
    #     self.node0.predict()
    #     node0_pred_full_state = self.node0.full_state_prediction
    #     node0_pred_full_cov = self.node0.full_cov_prediction
    #     node0_target_new_pos = self.node0.predicted_targets[0].state
    #
    #     self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
    #                        None])
    #     node0_update_full_state = self.node0.full_state_update
    #     node0_update_full_cov = self.node0.full_cov_update
    #     node0_target_update_pos = self.node0.updated_targets[0].state
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_target_new_pos, node0_target_update_pos)
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_pred_full_state, node0_update_full_state)
    #     assert_raises(AssertionError, assert_array_equal,
    #                   node0_pred_full_cov, node0_update_full_cov)
    #
    # def test_init_consensus(self):
    #     print('init consensus')
    #     self.node0.predict()
    #     self.node1.predict()
    #
    #     self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
    #                        np.array([[1.01], [1.01], [1.01], [1.01]])])
    #     self.node1.update([np.array([[-1.01], [1.01], [1.01], [1.01]]),
    #                        np.array([[-1.01], [1.01], [1.01], [1.01]])])
    #
    #     self.node0.init_consensus()
    #     self.node1.init_consensus()
    #
    #     assert self.node0.omega is not None
    #     assert self.node1.omega is not None
    #     assert self.node0.qs is not None
    #     assert self.node1.qs is not None
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   self.node0.omega, self.node1.omega)
    #     assert_raises(AssertionError, assert_array_equal,
    #                   self.node0.qs, self.node1.qs)
    #
    # def test_consensus_filter(self):
    #     print('consensus filter')
    #     self.node0.predict()
    #     self.node1.predict()
    #
    #     self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
    #                        np.array([[1.01], [1.01], [1.01], [1.01]])])
    #     self.node1.update([np.array([[-1.01], [1.01], [1.01], [1.01]]),
    #                        np.array([[-1.01], [1.01], [1.01], [1.01]])])
    #
    #     self.node0.init_consensus()
    #     self.node1.init_consensus()
    #
    #     prev_omega = deepcopy(self.node0.omega)
    #     prev_qs = deepcopy(self.node0.qs)
    #
    #     self.node0.consensus_filter([self.node1.omega],
    #                                 [self.node1.qs],
    #                                 [0.5])
    #     assert_raises(AssertionError, assert_array_equal,
    #                   prev_omega, self.node0.omega)
    #     assert_raises(AssertionError, assert_array_equal,
    #                   prev_qs, self.node0.qs)
    #
    def test_after_consensus_update(self):
        print('after consensus update')

        self.node0.predict()
        self.node1.predict()

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[1.01], [1.01], [1.01], [1.01]])])
        self.node1.update([np.array([[-1.01], [1.01], [1.01], [1.01]]),
                           np.array([[-1.01], [1.01], [1.01], [1.01]])])

        old_state = self.node0.full_state_update
        old_cov = self.node0.full_cov_update

        self.node0.init_consensus()
        self.node1.init_consensus()

        self.node0.consensus_filter([self.node1.omega],
                                    [self.node1.qs],
                                    [0.5])
        self.node0.after_consensus_update()

        assert_raises(AssertionError, assert_array_equal,
                      old_state, self.node0.full_state)
        assert_raises(AssertionError, assert_array_equal,
                      old_cov, self.node0.full_cov)

    def test_step(self):
        print('step')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[1], [1]]),
                      np.array([[1], [1]])]
                  }
        self.network.step_through(inputs)

        assert len(self.network.adjacencies) == len(inputs)




