from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from models import Birth, Clutter, Estimate, Measurement, Survival, Transition
from PHDFilter import PHDFilter
from PHDGenerator import PHDGenerator


class FilterTests(TestCase):

    def setUp(self):
        super().setUp()

        self.birth_model = Birth(1)
        self.clutter_model = Clutter(1)
        self.transition_model = Transition()
        self.measurement_model = Measurement(3, .98)
        self.survival_model = Survival()
        self.estimation_model = Estimate(1)

        init_targets = [np.array([[0.], [0.], [0.], [0.]]),
                        np.array([[10.], [10.], [0.], [0.]]),
                        np.array([[20.], [20.], [0.], [0.]])]

        self.filter = PHDFilter(birth_model=self.birth_model,
                                clutter_model=self.clutter_model,
                                measurement_model=self.measurement_model,
                                transition_model=self.transition_model,
                                survival_model=self.survival_model,
                                estimation_model=self.estimation_model,
                                init_targets=init_targets,
                                init_weights=[1./3, 1./3, 1./3])

        self.generator = PHDGenerator(birth_model=self.birth_model,
                                      clutter_model=self.clutter_model,
                                      transition_model=self.transition_model,
                                      measurement_model=self.measurement_model,
                                      timesteps=200,
                                      init_targets=init_targets)
        self.generator.generate(50)
        self.generator.plot_gif(show_clutter=True)

    # def tearDown(self):
    #     super().tearDown()
    #
    # def test_predict(self):
    #     self.filter.predict()
    #
    #     assert self.filter.predicted_pos[0].shape == \
    #            self.filter.current_targets[0].shape
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   self.filter.current_targets,
    #                   self.filter.predicted_pos)
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   self.filter.current_weights,
    #                   self.filter.predicted_weights)
    #
    # def test_update(self):
    #     self.filter.predict()
    #     self.filter.update(self.generator.observations[0])
    #
    #     assert len(self.filter.predicted_weights) == \
    #            len(self.filter.updated_weights)
    #     assert_raises(AssertionError, assert_array_equal,
    #                   self.filter.predicted_weights,
    #                   self.filter.updated_weights)
    #
    # def test_resample(self):
    #     self.filter.predict()
    #     self.filter.update(self.generator.observations[0])
    #     self.filter.resample()
    #
    #     assert len(self.filter.resampled_pos) == \
    #            len(self.filter.resampled_weights)
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   self.filter.updated_weights,
    #                   self.filter.resampled_weights)
    #
    # def test_estimate(self):
    #     self.filter.predict()
    #     self.filter.update(self.generator.observations[0])
    #     self.filter.resample()
    #     self.filter.estimate()
    #
    #     assert self.filter.centroids.shape[0] >= 1
    #     assert self.filter.centroids.shape[1] == 2
    #
    # def test_plot(self):
    #     self.filter.predict()
    #     self.filter.update(self.generator.observations[0])
    #     self.filter.resample()
    #     self.filter.estimate()
    #     self.filter.plot(0)

    def test_step_through(self):
        self.filter.predict()
        self.filter.step_through(self.generator.observations)
        self.filter.plot_centroids()

