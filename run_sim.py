import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from PHDFilterNetwork import PHDFilterNetwork
from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


np.random.seed(42)
"""
Generate Data
"""
generator = SimGenerator(5, init_targets=[Target()])
generator.generate(50)


"""
Birth Models for entire space
"""
birthgmm = []
for x in range(-50, 50, 10):
    for y in range(-50, 50, 10):
        target = Target(init_weight=1,
                        init_state=np.array([[x], [y], [0.0], [0.0]]),
                        dt_1=0, dt_2=0)
        birthgmm.append(target)

"""
Create Nodes
"""

filternode_1 = PHDFilterNode(0, birthgmm,
                             position=(-25, 0),
                             region=[(-45, -5), (-20, 20)])
filternode_2 = PHDFilterNode(1, birthgmm,
                             position=(0, 0),
                             region=[(-20, 20), (-20, 20)])
filternode_3 = PHDFilterNode(2, birthgmm,
                             position=(25, 0),
                             region=[(5, 45), (-20, 20)])

"""
Create Graph
"""

G = nx.Graph()
for i in range(0, 2):
    G.add_edge(i, i + 1)
node_attrs = {0: filternode_1,
              1: filternode_2,
              2: filternode_3}
weight_attrs = {0: 1.0 / 3, 1: 1.0 / 3, 2: 1.0 / 3}

filternetwork = PHDFilterNetwork(node_attrs, weight_attrs, G)


"""
Run Simulation
"""
filternetwork.step_through(generator.observations, how='arith')

"""
Plot
"""

for i, targets in generator.observations.items():
    # Plot True Positions

    ax = plt.axes()
    plt.xlim((-50, 50))
    plt.ylim((-50, 50))
    x = []
    y = []
    for t in targets:
        x.append(t[0])
        y.append(t[1])
    ax.scatter(x, y, label='targets')

    # Plot Robot Positions
    x = []
    y = []
    all_nodes = nx.get_node_attributes(filternetwork.network, 'node')
    for n, node in all_nodes.items():
        pos = node.position
        radius = (node.region[0][1] - node.region[0][0]) / 2.0
        p = plt.Circle(pos, radius, alpha=0.2, color='blue')
        ax.add_patch(p)

        # Plot Predicted Positions After Consensus
        for t in node.consensus_positions[i]:
            x.append(t[0][0])
            y.append(t[1][0])
    ax.scatter(x, y, label='estimates')

    plt.legend()
    plt.savefig('test/{i}.png'.format(i=i))
    plt.clf()


