from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from models import Birth, Transition, Measurement

# np.random.seed(4)


class ModelTests(TestCase):

    def setUp(self):
        super().setUp()

        self.birth_model = Birth(1)
        self.transition_model = Transition()
        self.measurement_model = Measurement(3, .98)

    def tearDown(self):
        super().tearDown()

    def test_birth(self):
        N, positions = self.birth_model.Sample()

        assert N >= 0
        assert len(positions) == N

    def test_transition(self):
        current_state = np.array([[0.], [0.], [0.], [0.]])
        next_state = self.transition_model.AdvanceState(current_state)

        assert_raises(AssertionError, assert_array_equal,
                      current_state, next_state)

    def test_detection_prob(self):
        targets = [np.array([[-150], [0.], [0.], [0.]]),
                   np.array([[0], [0.], [0.], [0.]])]
        probs = []
        for t in targets:
            probs.append(self.measurement_model.DetectionProbability(t))

        assert probs == [0, .98]








