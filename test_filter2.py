from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


class FilterTests(TestCase):

    def setUp(self):
        super().setUp()

        self.generator = SimGenerator(1, 0.2,
                                      init_targets=[Target()])
        self.node = PHDFilterNode(J=100)

        self.generator.generate(20)
        self.generator.plot_gif(show_clutter=True)

    def tearDown(self):
        super().tearDown()

    def test_predict(self):
        self.node.predict()

        assert self.node.predicted_pos[0].shape == (4, 1)

        assert_raises(AssertionError, assert_array_equal,
                      self.node.current_particles,
                      self.node.predicted_pos)

        assert_raises(AssertionError, assert_array_equal,
                      self.node.current_weights,
                      self.node.predicted_weights)

    def test_update(self):
        self.node.predict()
        self.node.update(self.generator.observations[0])

        assert len(self.node.predicted_weights) == \
               len(self.node.updated_weights)
        assert_raises(AssertionError, assert_array_equal,
                      self.node.predicted_weights,
                      self.node.updated_weights)

    def test_resample(self):
        self.node.predict()
        self.node.update(self.generator.observations[0])
        self.node.resample()

        assert len(self.node.resampled_pos) == \
               len(self.node.resampled_weights)

        assert_raises(AssertionError, assert_array_equal,
                      self.node.updated_weights,
                      self.node.resampled_weights)

    def test_estimate(self):
        self.node.predict()
        self.node.update(self.generator.observations[0])
        self.node.resample()
        self.node.estimate()
        print(self.node.centroids)

        assert self.node.centroids.shape[0] >= 1
        assert self.node.centroids.shape[1] == 2

    def test_step_through(self):
        self.node.step_through(self.generator.observations)
        self.node.plot_centroids()

