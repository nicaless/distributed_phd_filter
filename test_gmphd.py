from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from SimGenerator import SimGenerator
from target import Target


class GMPHDTest(TestCase):

    def setUp(self):
        super().setUp()

        self.target = Target()

        self.generator = SimGenerator(5, 0.2, init_targets=[Target()])
        self.generator.generate(20)
        self.generator.plot(show_clutter=True)

        # self.birthgmm = []
        # for x in range(-50, 50, 5):
        #     for y in range(-50, 50, 5):
        #         target = Target(init_state=np.array([[x], [y], [0.0], [0.0]]))
        #         self.birthgmm.append(target)

    def tearDown(self):
        super().tearDown()

    def test_target_next_state(self):
        init_state = self.target.state
        init_cov = self.target.cov
        self.target.next_state()
        next_state = self.target.state
        next_cov = self.target.cov

        assert_raises(AssertionError, assert_array_equal,
                      init_state, next_state)
        assert_raises(AssertionError, assert_array_equal,
                      init_cov, next_cov)
        assert len(self.target.all_states) == 2
        assert len(self.target.all_cov) == 2

    def test_target_get_measurement(self):
        self.target.next_state()
        next_state = self.target.state
        obs = self.target.get_measurement()
        next_state_pos = np.array([[next_state[0][0]],
                                   [next_state[1][0]]])
        assert_array_equal(obs, next_state_pos)

    def test_target_sample(self):
        first_sample = self.target.sample()
        self.target.next_state()
        second_sample = self.target.sample()

        assert first_sample.shape == second_sample.shape
