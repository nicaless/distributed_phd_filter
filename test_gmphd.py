from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


class GMPHDTest(TestCase):

    def setUp(self):
        super().setUp()

        self.target = Target()

        # self.generator = SimGenerator(5, 0.2, init_targets=[Target()])
        self.generator = SimGenerator(5, init_targets=[Target()])
        self.generator.generate(20)
        self.generator.plot(show_clutter=True)

        self.birthgmm = []
        for x in range(-50, 50, 10):
            for y in range(-50, 50, 10):
                target = Target(init_weight=0.05,
                                init_state=np.array([[x], [y], [0.1], [0.1]]),
                                dt_1=0, dt_2=0)
                self.birthgmm.append(target)
        # self.region = [(-50, 50), (-50, 50)]
        # birth0 = Target(
        #     init_weight=0.05,
        #     init_state=np.array([
        #         [self.region[0][0]],
        #         [self.region[1][0]],
        #         [0.1], [0.1]]))
        # birth1 = Target(
        #     init_weight=0.05,
        #     init_state=np.array([
        #         [self.region[0][0]],
        #         [self.region[1][1]],
        #         [0.1], [0.1]]),
        #     dt_2=-1)
        # birth2 = Target(
        #     init_weight=0.05,
        #     init_state=np.array(
        #         [[self.region[0][1]],
        #          [self.region[1][1]],
        #          [0.1], [0.1]]),
        #     dt_1=-1, dt_2=-1)
        # birth3 = Target(
        #     init_weight=0.05,
        #     init_state=np.array(
        #         [[self.region[0][1]],
        #          [self.region[1][1]],
        #          [0.1], [0.1]]),
        #     dt_1=-1)
        # self.birthgmm = [birth0, birth1, birth2, birth3]

        self.filternode = PHDFilterNode(0, self.birthgmm)

    def tearDown(self):
        super().tearDown()

    # def test_target_next_state(self):
    #     init_state = self.target.state
    #     init_cov = self.target.state_cov
    #     self.target.next_state()
    #     next_state = self.target.state
    #     next_cov = self.target.state_cov
    #
    #     assert_raises(AssertionError, assert_array_equal,
    #                   init_state, next_state)
    #     assert_raises(AssertionError, assert_array_equal,
    #                   init_cov, next_cov)
    #     assert len(self.target.all_states) == 2
    #     assert len(self.target.all_cov) == 2
    #
    # def test_target_get_measurement(self):
    #     self.target.next_state()
    #     next_state = self.target.state
    #     obs = self.target.get_measurement()
    #     next_state_pos = np.array([[next_state[0][0]],
    #                                [next_state[1][0]]])
    #     assert_array_equal(obs, next_state_pos)
    #
    # def test_target_sample(self):
    #     first_sample = self.target.sample()
    #     self.target.next_state()
    #     second_sample = self.target.sample()
    #
    #     assert first_sample.shape == second_sample.shape
    #
    # def test_node_predict(self):
    #     self.filternode.predict()
    #
    #     assert len(self.filternode.predicted_pos) == \
    #            len(self.filternode.predicted_targets)
    #
    # def test_node_update(self):
    #     self.filternode.predict()
    #     self.filternode.update(self.generator.observations[0])
    #
    #     assert len(self.filternode.updated_targets) >= \
    #            len(self.filternode.predicted_targets)
    #
    # def test_node_prune(self):
    #     self.filternode.predict()
    #     self.filternode.update(self.generator.observations[0])
    #     self.filternode.prune()
    #
    #     assert len(self.filternode.pruned_targets) <= \
    #            len(self.filternode.updated_targets)
    #
    # def test_node_merge(self):
    #     self.filternode.predict()
    #     self.filternode.update(self.generator.observations[0])
    #     self.filternode.prune()
    #     self.filternode.merge()
    #     # print(len(self.generator.observations[0]))
    #     # print(len(self.filternode.merged_targets))
    #     # print(len(self.filternode.pruned_targets))
    #     # print(sum([t.weight for t in self.filternode.merged_targets]))
    #
    #     assert len(self.filternode.merged_targets) <= \
    #            len(self.filternode.pruned_targets)

    # def test_node_extract(self):
    #     self.filternode.predict()
    #     self.filternode.update(self.generator.observations[0])
    #     self.filternode.prune()
    #     self.filternode.merge()
    #     self.filternode.targets = self.filternode.merged_targets
    #     x, y = self.filternode.extractstates()
    #     print(x)
    #     print(y)

    def test_node_step_through(self):
        self.filternode.step_through(self.generator.observations)
        print(sum([t.weight for t in self.filternode.targets]))




