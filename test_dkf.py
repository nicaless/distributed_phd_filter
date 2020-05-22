from unittest import TestCase

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

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

        self.node0 = DKFNode(1, [deepcopy(self.target1),
                                 deepcopy(self.target2),
                                 ])

    def tearDown(self):
        super().tearDown()

    def test_01_predict(self):
        print('predict')

        inputs = [np.array([[1], [1]]), np.array([[1], [1]])]

        node0_target_curr_pos = self.node0.targets[0].state

        self.node0.predict(inputs)

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

        inputs = [np.array([[1], [1]]), np.array([[1], [1]])]

        self.node0.predict(inputs)
        node0_pred_full_state = self.node0.full_state_prediction
        node0_pred_full_cov = self.node0.full_cov_prediction
        node0_target_new_pos = self.node0.predicted_targets[0].state

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           np.array([[1.01], [1.01], [1.01], [1.01]])])
        node0_update_full_state = self.node0.full_state_update
        node0_update_full_cov = self.node0.full_cov_update
        node0_target_update_pos = self.node0.updated_targets[0].state

        assert_raises(AssertionError, assert_array_equal,
                      node0_target_new_pos, node0_target_update_pos)

        assert_raises(AssertionError, assert_array_equal,
                      node0_pred_full_state, node0_update_full_state)
        assert_raises(AssertionError, assert_array_equal,
                      node0_pred_full_cov, node0_update_full_cov)

    def test_03_update_missing_measurement(self):
        print('update, missing measurement')

        inputs = [np.array([[1], [1]]), np.array([[1], [1]])]

        self.node0.predict(inputs)
        node0_pred_full_state = self.node0.full_state_prediction
        node0_pred_full_cov = self.node0.full_cov_prediction
        node0_target_new_pos = self.node0.predicted_targets[0].state

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]]),
                           None])
        node0_update_full_state = self.node0.full_state_update
        node0_update_full_cov = self.node0.full_cov_update
        node0_target_update_pos = self.node0.updated_targets[0].state

        assert_raises(AssertionError, assert_array_equal,
                      node0_target_new_pos, node0_target_update_pos)

        assert_raises(AssertionError, assert_array_equal,
                      node0_pred_full_state, node0_update_full_state)
        assert_raises(AssertionError, assert_array_equal,
                      node0_pred_full_cov, node0_update_full_cov)

    def test_step(self):
        print('step')


