from unittest import TestCase

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

from PHDFilterNetwork import PHDFilterNetwork
from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


region = [(-50, 50), (-50, 50)]

corner0 = Target(init_state=np.array([[region[0][0] + 10],
                                      [region[1][0] + 10],
                                      [0.1], [0.1]]))
corner1 = Target(init_state=np.array([[region[0][0] + 10],
                                      [region[1][1] - 10],
                                      [0.1], [0.1]]), dt_2=-1)
corner2 = Target(init_state=np.array([[region[0][1] - 10],
                                      [region[1][1] - 10],
                                      [0.1], [0.1]]), dt_1=-1, dt_2=-1)
corner3 = Target(init_state=np.array([[region[0][1] - 10],
                                      [region[1][0] + 10],
                                      [0.1], [0.1]]), dt_1=-1)
birthgmm = [corner0, corner1, corner2, corner3]


class FilterNodeTests(TestCase):

    def setUp(self):
        super().setUp()

        np.random.seed(42)
        self.N = 20
        self.generator = SimGenerator()
        self.generator.generate(steps=self.N)
        self.generator.plot()

        self.node_full = PHDFilterNode(0, birthgmm)

        self.node_attrs = {}
        num_nodes = 3
        x_start = -50 + (100.0 / (num_nodes + 1))
        pos_start = np.array([x_start, 0, 20])
        pos_init_dist = np.floor(100.0 / (num_nodes + 1))
        fov = 20
        for n in range(num_nodes):
            pos = pos_start + np.array([n * pos_init_dist, 0, 0])
            region = [(pos[0] - fov, pos[0] + fov),
                      (pos[1] - fov, pos[1] + fov)]
            self.node_attrs[n] = PHDFilterNode(n, birthgmm,
                                               position=pos,
                                               region=region)
        G = nx.Graph()
        for i in range(num_nodes - 1):
            G.add_edge(i, i + 1)

        weight_attrs = {}
        for i in range(num_nodes):
            weight_attrs[i] = {}
            self_degree = G.degree(i)
            metropolis_weights = []
            for n in G.neighbors(i):
                degree = G.degree(n)
                mw = 1 / (1 + max(self_degree, degree))
                weight_attrs[i][n] = mw
                metropolis_weights.append(mw)
            weight_attrs[i][i] = 1 - sum(metropolis_weights)

        self.filternetwork = PHDFilterNetwork(self.node_attrs,
                                              weight_attrs, G)

    def tearDown(self):
        super().tearDown()

    def test_local_phd(self):
        print('checking local phds of individual nodes')
        nodes = nx.get_node_attributes(self.filternetwork.network, 'node')

        for id, n in nodes.items():
            n.step_through(self.generator.observations)

        # for i in range(10, self.N):
        #     for id, n in nodes.items():
        #         print(id, i)
        #         assert len(n.preconsensus_positions[i]) > 4

        # Plot all node estimates
        for i, pos in self.generator.observations.items():
            ax = plt.axes()
            plt.xlim((-50, 50))
            plt.ylim((-50, 50))

            node_estimates = {}
            x = []
            y = []
            for p in pos:
                x.append(p[0])
                y.append(p[1])
            node_estimates[-1] = {'x': x, 'y': y}

            for id, n in nodes.items():
                node_sub = n
                x = []
                y = []
                for ests in node_sub.preconsensus_positions[i]:
                    x.append(ests[0][0])
                    y.append(ests[1][0])
                node_estimates[id] = {'x': x, 'y': y}

            for key, vals in node_estimates.items():
                if key == -1:
                    plt.scatter(vals['x'], vals['y'], label=key, alpha=0.5, s=20)
                else:
                    plt.scatter(vals['x'], vals['y'], label=key, alpha=0.5, s=10)

            # Plot FOV
            for id, n in nodes.items():
                p = plt.Circle((n.node_positions[i][0],
                                n.node_positions[i][1]),
                               20, alpha=0.1)
                ax.add_patch(p)

            plt.legend()
            plt.savefig('results_network/{i}.png'.format(i=i))
            plt.clf()

    def test_failure(self):
        pass

    def test_fuse_components(self):
        pass

    def test_cardinality_consensus(self):
        pass
