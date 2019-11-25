from unittest import TestCase

import numpy as np

from models import Birth, Clutter, Measurement, Survival, Transition
from PHDGenerator import PHDGenerator


class GeneratorTests(TestCase):

    def setUp(self):
        super().setUp()

        self.birth_model = Birth(1)
        self.clutter_model = Clutter(1)
        self.transition_model = Transition()
        self.measurement_model = Measurement(3, .98)

        init_targets = [np.array([[0.], [0.], [0.], [0.]]),
                        np.array([[10.], [10.], [0.], [0.]]),
                        np.array([[20.], [20.], [0.], [0.]])]

        self.generator = PHDGenerator(birth_model=self.birth_model,
                                      clutter_model=self.clutter_model,
                                      transition_model=self.transition_model,
                                      measurement_model=self.measurement_model,
                                      timesteps=200,
                                      init_targets=init_targets)

    def tearDown(self):
        super().tearDown()

    def test_iterate(self):
        self.generator.iterate(0)

        assert len(self.generator.true_targets.keys()) == 1
        assert len(self.generator.true_targets[0]) == \
               len(self.generator.last_timestep_targets)
        assert len(self.generator.true_targets[0]) == \
               len(self.generator.true_observations[0])
        assert (len(self.generator.true_observations[0]) +
                len(self.generator.clutter_observations[0])) == \
               len(self.generator.observations[0])

    def test_generate(self):
        self.generator.generate(10)

        assert len(self.generator.observations.keys()) == 10
        assert len(self.generator.true_targets.keys()) == 10

        self.generator.plot_gif(show_clutter=True)

    def test_reset(self):
        self.generator.generate(10)
        self.generator.reset()
        assert len(self.generator.observations.keys()) == 0
        assert len(self.generator.true_targets.keys()) == 0
