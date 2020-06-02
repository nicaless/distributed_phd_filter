import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

from ospa import *

from DKFNetwork import DKFNetwork
from DKFNode import DKFNode
from target import Target

np.random.seed(42)

"""
Params
"""
parser = argparse.ArgumentParser()
parser.add_argument('num_nodes', type=int, default=4)
parser.add_argument('num_targets', type=int, default=4)
parser.add_argument('run_name', default='4_nodes_test')
args = parser.parse_args()

num_nodes = args.num_nodes
num_targets = args.num_targets
run_name = args.run_name

fail_int = [5, 10, 15, 20]
x_start = -50 + (100.0 / (num_nodes + 1))
pos_start = np.array([x_start, 0, 20])
pos_init_dist = np.floor(100.0 / (num_nodes + 1))
fov = 30  # radius of FOV
noise_mult = [1.]


"""
Create Folder for Run
"""
if not os.path.exists(run_name):
    os.makedirs(run_name)
    os.makedirs(run_name + '/fail_sequence')


"""
Create Targets
"""
t1 = Target(init_state=np.array([[25], [25], [1.], [1.]]))
t2 = Target(init_state=np.array([[-25], [25], [1.], [1.]]))
t3 = Target(init_state=np.array([[-25], [-25], [1.], [1.]]))
t4 = Target(init_state=np.array([[25], [-25], [1.], [1.]]))
targets = [t1, t2, t3, t4]
# targets = []
# for t in range(num_targets):
#     targets.append(Target())


"""
Generate Data
"""
inputs = {}
for i in range(25):
    ins = []
    for t in range(num_targets):
        x_dir = np.random.choice([-1, 0, 1])
        y_dir = np.random.choice([-1, 0, 1])
        ins.append(np.array([[x_dir], [y_dir]]))
    inputs[i] = ins

"""
Create Nodes
"""
node_attrs = {}

for n in range(num_nodes):
    pos = pos_start + np.array([n*pos_init_dist, 0, 0])
    region = [(pos[0] - fov, pos[0] + fov),
              (pos[1] - fov, pos[1] + fov)]

    node_attrs[n] = DKFNode(n,
                            [deepcopy(t) for t in targets],
                            position=pos,
                            region=region)

"""
Create Graph
"""

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

"""
For Loop for all Simulations
"""
saved_fail_sequence = None
for noise in range(len(noise_mult)):
    # for opt in ['base', 'agent', 'team', 'greedy', 'random']:
    for opt in ['base', 'agent']:
        trial_name = run_name + '/{noise}_{o}'.format(noise=noise,
                                                      o=opt)
        print(trial_name)

        filternetwork = DKFNetwork(deepcopy(node_attrs),
                                   deepcopy(weight_attrs),
                                   deepcopy(G),
                                   [deepcopy(t) for t in targets])

        """
        Run Simulation
        """
        base = opt == 'base'
        if base:
            filternetwork.step_through(inputs,
                                       opt=opt,
                                       fail_int=fail_int,
                                       base=base,
                                       noise_mult=noise_mult[noise])

            """
            Save Fail Sequence
            """
            rpd_folder = run_name + '/fail_sequence'
            for i, vals in filternetwork.failures.items():
                rpd_filename = rpd_folder + '/{i}.csv'.format(i=i)
                np.savetxt(rpd_filename, vals[1], delimiter=",")
            df = pd.DataFrame.from_dict(filternetwork.failures, orient='index')
            df[[0]].to_csv(rpd_folder + '/_node_list.csv', header=None)
            saved_fail_sequence = filternetwork.failures
        else:
            filternetwork.step_through(inputs,
                                       opt=opt,
                                       fail_int=saved_fail_sequence,
                                       base=base,
                                       noise_mult=noise_mult[noise])

        """
        Save Data
        """
        if not os.path.exists(trial_name):
            os.makedirs(trial_name)
            os.makedirs(trial_name + '/3ds')
            os.makedirs(trial_name + '/overhead')
            os.makedirs(trial_name + '/topologies')
        filternetwork.save_metrics(trial_name)
        filternetwork.save_estimates(trial_name)
        filternetwork.save_positions(trial_name)
        filternetwork.save_true_target_states(trial_name)
        filternetwork.save_topologies(trial_name + '/topologies')


# TODO: covert to be series of overhead plots to include tracker positions
print('plot')
# Plot Targets
est = pd.read_csv('4_nodes_test/0_greedy/estimates.csv')
# est = est[est['time'] < 17]
colors = ['red', 'blue', 'green', 'orange']
for t in range(num_targets):
    df = pd.read_csv('4_nodes_test/0_base/target_{t}_positions.csv'.format(t=t))
    # df = df.loc[0:16]
    print(len(df))
    plt.plot(df['x'].values, df['y'].values, '+',
             label="True Target {t}".format(t=t), alpha=0.8, color=colors[t])

    e = est[est['target'] == t]
    e = e.groupby('time').agg({'x': 'mean', 'y': 'mean'}).reset_index()
    print(len(e))
    plt.plot(e['x'].values, e['y'].values, '--',
             label="Estimate {t}".format(t=t), alpha=0.5, color=colors[t])

plt.legend()
plt.savefig('test.png')

# TODO: plot coverage quality
# TODO: plot errors (are these errors calculated correctly)
# TODO: plot covariance





