from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from models import Birth, Clutter, Transition, Measurement
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

    def test_generate(self):
        print(self.generator.last_timestep_targets)
        self.generator.generate(0)

        print(self.generator.true_targets)
        print(self.generator.observations)

        assert len(self.generator.true_targets.keys()) == 1
        assert len(self.generator.true_targets[0]) == \
               len(self.generator.last_timestep_targets)
        assert len(self.generator.true_targets[0]) <= \
               len(self.generator.observations[0])
