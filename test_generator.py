from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from SimGenerator import SimGenerator

np.random.seed(42)


class SimGeneratorTests(TestCase):

    def setUp(self):
        super().setUp()

        self.N = 20
        self.generator = SimGenerator()
        self.generator.generate(steps=self.N)
        self.generator.plot()

    def tearDown(self):
        super().tearDown()

    def test_len_targets(self):
        assert len(self.generator.targets) == self.N
