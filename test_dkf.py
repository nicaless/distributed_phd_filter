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

    def test_01_predict(self):
        print('predict')

        node0_target_curr_pos = self.node0.targets[0].state
        self.node0.predict(1)

        node0_pred_full_state = self.node0.full_state_prediction
        node0_pred_full_cov = self.node0.full_cov_prediction
        node0_pred_pos = self.node0.predicted_pos[0]
        node0_target_new_pos = self.node0.predicted_targets[0].state

        assert node0_pred_full_state is not None
        assert node0_pred_full_cov is not None

        assert_raises(AssertionError, assert_array_equal,
                      node0_target_curr_pos, node0_target_new_pos)

        assert_array_equal(node0_target_new_pos, node0_pred_pos)

    def test_02_update(self):
        print('update')

        self.node0.predict(1)
        node0_pred_full_cov = self.node0.full_cov_prediction

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[1.01], [1.01], [1.01], [1.01]])])

        node0_update_full_cov = self.node0.full_cov_update

        assert_raises(AssertionError, assert_array_equal,
                      node0_pred_full_cov, node0_update_full_cov)

    def test_03_update_missing_measurement(self):
        print('update, missing measurement')

        self.node0.predict(1)
        node0_pred_full_cov = self.node0.full_cov_prediction

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           None])
        node0_update_full_cov = self.node0.full_cov_update

        assert_raises(AssertionError, assert_array_equal,
                      node0_pred_full_cov, node0_update_full_cov)

    def test_04_init_consensus(self):
        print('init consensus')
        self.node0.predict(2)
        self.node1.predict(2)

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[1.01], [1.01], [1.01], [1.01]])])
        self.node1.update([np.array([[-1.01], [1.01], [1.01], [1.01]]),
                           np.array([[-1.01], [1.01], [1.01], [1.01]])])

        self.node0.init_consensus()
        self.node1.init_consensus()

        assert self.node0.omega is not None
        assert self.node1.omega is not None
        assert self.node0.qs is not None
        assert self.node1.qs is not None

    def test_05_consensus_filter(self):
        print('consensus filter')
        self.node0.predict(2)
        self.node1.predict(2)

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[1.01], [1.01], [1.01], [1.01]])])
        self.node1.update([np.array([[-1.01], [1.01], [1.01], [1.01]]),
                           np.array([[-1.01], [1.01], [1.01], [1.01]])])

        self.node0.init_consensus()
        self.node1.init_consensus()

        prev_omega = deepcopy(self.node0.omega)
        prev_qs = deepcopy(self.node0.qs)

        self.node0.consensus_filter([self.node1.omega],
                                    [self.node1.qs],
                                    [0.5])
        assert_raises(AssertionError, assert_array_equal,
                      prev_omega, self.node0.omega)
        assert_raises(AssertionError, assert_array_equal,
                      prev_qs, self.node0.qs)

    def test_06_after_consensus_update(self):
        print('after consensus update')

        self.node0.predict(2)
        self.node1.predict(2)

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[-1.01], [1.01], [1.01], [1.01]])])
        self.node1.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[-1.01], [1.01], [1.01], [1.01]])])

        old_state = self.node0.full_state_prediction
        old_cov = self.node0.full_cov_update

        self.node0.init_consensus()
        self.node1.init_consensus()

        self.node0.consensus_filter([self.node1.omega],
                                    [self.node1.qs],
                                    [0.5])
        self.node0.intermediate_cov_update()
        self.node0.after_consensus_update(2)

        assert_raises(AssertionError, assert_array_equal,
                      old_state, self.node0.full_state)
        assert_raises(AssertionError, assert_array_equal,
                      old_cov, self.node0.full_cov)

    def test_06_after_consensus_update_missing_measurement(self):
        print('after consensus update missing measurement')

        self.node0.predict(2)
        self.node1.predict(2)

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           None])
        self.node1.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[-1.01], [1.01], [1.01], [1.01]])])

        old_state = self.node0.full_state_prediction
        old_cov = self.node0.full_cov_update

        self.node0.init_consensus()
        self.node1.init_consensus()

        self.node0.consensus_filter([self.node1.omega],
                                    [self.node1.qs],
                                    [0.5])
        self.node0.intermediate_cov_update()
        self.node0.after_consensus_update(2)

        assert_raises(AssertionError, assert_array_equal,
                      old_state, self.node0.full_state)
        assert_raises(AssertionError, assert_array_equal,
                      old_cov, self.node0.full_cov)

    def test_07_step(self):
        print('step')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements)

        assert len(self.network.adjacencies) == len(inputs)
        assert len(self.network.errors) == len(inputs)
        assert len(self.network.mean_trace_cov) == len(inputs)

    def test_08_save_metrics(self):
        print('save metrics')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements)
        self.network.save_metrics('~')

    def test_09_save_estimates(self):
        print('save estimates')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements)
        self.network.save_estimates('~')

    def test_10_step_with_failure_agent(self):
        print('step with failure - agent')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])],
                  3: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  4: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  5: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements, fail_int=[1])
        self.network.save_metrics('~')
        self.network.save_estimates('~')

        assert len(self.network.adjacencies) == len(inputs)
        assert len(self.network.errors) == len(inputs)
        assert len(self.network.mean_trace_cov) == len(inputs)

    def test_11_step_with_failure_team(self):
        print('step with failure - team')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])],
                  3: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  4: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  5: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements,
                                  opt='team', fail_int=[1])

        assert len(self.network.adjacencies) == len(inputs)
        assert len(self.network.errors) == len(inputs)
        assert len(self.network.mean_trace_cov) == len(inputs)

    def test_12_step_with_failure_greedy(self):
        print('step with failure - greedy')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])],
                  3: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  4: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  5: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements,
                                  opt='greedy', fail_int=[1])

        assert len(self.network.adjacencies) == len(inputs)
        assert len(self.network.errors) == len(inputs)
        assert len(self.network.mean_trace_cov) == len(inputs)

    def test_13_step_with_failure_random(self):
        print('step with failure - random')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])],
                  3: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  4: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  5: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        measurements = {0: [np.array([[1.01], [1.01], [2.01], [2.01]]),
                            np.array([[1.01], [1.01], [2.01], [2.01]])],
                        1: [np.array([[3.01], [3.01], [2.01], [2.01]]),
                            np.array([[3.01], [3.01], [2.01], [2.01]])],
                        2: [np.array([[5.01], [5.01], [1.01], [1.01]]),
                            np.array([[5.01], [5.01], [1.01], [1.01]])]
                        }
        self.network.step_through(inputs, measurements=measurements,
                                  opt='random', fail_int=[1])

        assert len(self.network.adjacencies) == len(inputs)
        assert len(self.network.errors) == len(inputs)
        assert len(self.network.mean_trace_cov) == len(inputs)

    def test_14_save_target_states(self):
        print('save target_states')

        inputs = {0: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  1: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  2: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])],
                  3: [np.array([[1], [1]]),
                      np.array([[1], [1]])],
                  4: [np.array([[0], [0]]),
                      np.array([[0], [0]])],
                  5: [np.array([[-1], [-1]]),
                      np.array([[-1], [-1]])]
                  }
        self.network.step_through(inputs)
        self.network.save_true_target_states('~')


