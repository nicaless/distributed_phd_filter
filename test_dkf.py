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

        self.target = Target()

        self.node0 = DKFNode(1, deepcopy(self.target))

    def tearDown(self):
        super().tearDown()

    def test_01_predict(self):
        print('predict')

        input = np.array([[1], [1]])

        node0_target_curr_pos = self.node0.targets[0].state
        self.node0.predict(input)
        node0_pred_pos = self.node0.predicted_pos[0]
        node0_target_new_pos = self.node0.predicted_targets[0].state

        assert_raises(AssertionError, assert_array_equal,
                      node0_target_curr_pos, node0_target_new_pos)

        assert_array_equal(node0_target_new_pos, node0_pred_pos)

        print(node0_target_curr_pos)
        print(node0_target_new_pos)

    def test_02_update(self):
        print('update')

        input = np.array([[1], [1]])

        self.node0.predict(input)
        node0_target_new_pos = self.node0.predicted_targets[0].state

        self.node0.update([np.array([[1.01], [1.01], [1.01], [1.01]])])
        node0_target_update_pos = self.node0.updated_targets[0].state

        assert_raises(AssertionError, assert_array_equal,
                      node0_target_new_pos, node0_target_update_pos)

