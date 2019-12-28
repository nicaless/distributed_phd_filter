from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from PHDFilterNetwork import PHDFilterNetwork
from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


np.random.seed(42)
"""
Generate Data
"""
fail_int = [i for i in range(50) if i == 10]
generator = SimGenerator(5, init_targets=[Target()])
generator.generate(50)


"""
Birth Models for entire space
"""
birthgmm = []
for x in range(-50, 50, 5):
    for y in range(-50, 50, 5):
        target = Target(init_weight=1,
                        init_state=np.array([[x], [y], [0.0], [0.0]]),
                        dt_1=0, dt_2=0)
        birthgmm.append(target)

"""
Create Nodes
"""

filternode_1 = PHDFilterNode(0, birthgmm,
                             position=np.array([-25, 0, 20]),
                             region=[(-45, -5), (-20, 20)])
filternode_2 = PHDFilterNode(1, birthgmm,
                             position=np.array([0, 0, 20]),
                             region=[(-20, 20), (-20, 20)])
filternode_3 = PHDFilterNode(2, birthgmm,
                             position=np.array([25, 0, 20]),
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

weight_attrs = {}
for i in range(0, 3):
    weight_attrs[i] = {}
    self_degree = G.degree(i)
    metropolis_weights = []
    for n in G.neighbors(i):
        degree = G.degree(n)
        mw = 1 / (1 + max(self_degree, degree))
        weight_attrs[i][n] = mw
        metropolis_weights.append(mw)
    weight_attrs[i][i] = 1 - sum(metropolis_weights)


filternetwork = PHDFilterNetwork(node_attrs, weight_attrs, G)
filternetwork2 = PHDFilterNetwork(deepcopy(node_attrs),
                                  deepcopy(weight_attrs),
                                  deepcopy(G))

"""
Run Simulation
"""
filternetwork.step_through(generator.observations,
                           generator.true_positions,
                           how='geom',
                           opt='greedy',
                           fail_int=fail_int)

# filternetwork2.step_through(generator.observations,
#                             generator.true_positions,
#                             how='arith',
#                             opt='agent',
#                             fail_sequence=filternetwork.failures)

"""
Plot Positions
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
        # pos = node.position
        pos = node.node_positions[i]
        radius = node.fov
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

"""
Plot Errors, Covariance
"""
time = []
error = []
max_trace_cov = []
for t in generator.observations.keys():
    time.append(t)
    error.append(filternetwork.errors[t])
    max_trace_cov.append(filternetwork.max_trace_cov[t])

plt.plot(time, error, label='error')
plt.legend()
plt.savefig('test/_errors.png')
plt.clf()
plt.plot(time, max_trace_cov, label='max_tr_cov')
plt.legend()
plt.savefig('test/_max_tr_cov.png')
plt.clf()


"""
Save Fail Sequence
"""

for i, vals in filternetwork.failures.items():
    rpd_filename = 'fail_sequence/' + '{i}.csv'.format(i=i)
    np.savetxt(rpd_filename, vals[1], delimiter=",")
df = pd.DataFrame.from_dict(filternetwork.failures, orient='index')
df[[0]].to_csv('fail_sequence/_node_list.csv', header=None)

"""
Save Adjacencies
"""
# print(filternetwork.adjacencies)
# print(filternetwork.failures)

