from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from models import Birth, Measurement, Transition, Resample

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

    def test_birth_weight(self):
        N, positions = self.birth_model.Sample()
        weights = self.birth_model.Weight(N)

        assert N == len(weights)
        if N > 0:
            assert weights[0] == 1.0 / N

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

    def test_resample(self):
        weights = [.2, .2, .2, .00001]
        particle_mass = np.sum(weights)
        indices = Resample(np.array(weights) / particle_mass)

        assert len(indices) <= len(weights)










