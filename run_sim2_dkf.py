import argparse
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os

from DKFNetwork import DKFNetwork
from DKFNode import DKFNode
from target import Target

import numpy as np
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

# fail_int = [5, 10, 15, 20]
total_time_steps = 50
fails_before_saturation = num_nodes * (num_nodes - 1) / 2 - (num_nodes - 1)
fail_freq = int(np.ceil(total_time_steps / fails_before_saturation))
# fail_int = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # time steps at which failure occurs
fail_int = list(range(1, total_time_steps, fail_freq))
print(fail_int)
x_start = -50 + (100.0 / (num_nodes + 1))
pos_start = np.array([x_start, 0, 20])
pos_init_dist = np.floor(100.0 / (num_nodes + 1))
fov = 30  # radius of FOV
noise_mult = [3., 3., 3., 3., 3., 3., 3., 3., 3., 3.]


"""
Create Folder for Run
"""
if not os.path.exists(run_name):
    os.makedirs(run_name)
    os.makedirs(run_name + '/fail_sequence')


"""
Create Targets
"""
t1 = Target(init_state=np.array([[10], [10], [1.], [1.]]))
t2 = Target(init_state=np.array([[-10], [10], [1.], [1.]]))
t3 = Target(init_state=np.array([[-10], [-10], [1.], [1.]]))
t4 = Target(init_state=np.array([[10], [-10], [1.], [1.]]))
targets = [t1, t2, t3, t4]
# targets = []
# for t in range(num_targets):
#     targets.append(Target())


"""
Generate Data
"""
inputs = {}
for i in range(total_time_steps):
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
    for opt in ['base', 'agent', 'greedy']:
    # for opt in ['base', 'agent']:
    # for opt in ['base', 'team']:
    # for opt in ['base']:
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


# print('plot')
# # Plot Targets
# for opt in ['agent', 'team', 'greedy', 'random']:
#     est = pd.read_csv('4_nodes_test/0_{opt}/estimates.csv'.format(opt=opt))
#     robot_pos = pd.read_csv('4_nodes_test/0_{opt}/robot_positions.csv'.format(opt=opt))
#     colors = ['red', 'blue', 'green', 'orange']
#     for i in [5, 10, 15, 20, 24]:
#         ax = plt.axes()
#         rs = robot_pos[robot_pos['time'] == i]
#         for t in range(num_targets):
#             df = pd.read_csv('4_nodes_test/0_{opt}/target_{t}_positions.csv'.format(opt=opt, t=t))
#             tmp = df.loc[i - 5 + 1: i + 2]

#             plt.plot(tmp['x'].values, tmp['y'].values, '+',
#                      label="True Target {t}".format(t=t), alpha=0.8, color=colors[t])

#             e = est[est['target'] == t]
#             e = e.groupby('time').agg({'x': 'mean', 'y': 'mean'}).reset_index()
#             e = e[(e['time'] >= i - 5) & (e['time'] <= i+1)]
#             plt.plot(e['x'].values, e['y'].values, '--',
#                      label="Estimate {t}".format(t=t), alpha=0.5, color=colors[t])

#         plt.scatter(rs['x'].values, rs['y'].values, color='black', marker='x')

#         # Plot Adjacencies
#         edge_list = []
#         new_A = []
#         topology_file = '4_nodes_test/0_{opt}/topologies/{i}.csv'.format(opt=opt, i=i)
#         with open(topology_file, 'r') as f:
#             readCSV = csv.reader(f, delimiter=',')
#             for row in readCSV:
#                 data = list(map(float, row))
#                 new_A.append(data)
#         new_A = np.array(new_A)
#         num_drones = new_A.shape[0]
#         for n in range(num_drones):
#             for o in range(n+1, num_drones):
#                 if new_A[n, o] == 1:
#                     n_pos = rs[rs['node_id'] == n]
#                     o_pos = rs[rs['node_id'] == o]

#                     xl = [n_pos['x'].values[0], o_pos['x'].values[0]]
#                     yl = [n_pos['y'].values[0], o_pos['y'].values[0]]
#                     plt.plot(xl, yl, color='gray', alpha=0.5)

#             # Plot FOV
#             tmp_rs = rs[rs['node_id'] == n]
#             p = plt.Circle((tmp_rs['x'].values[0], tmp_rs['y'].values[0]), tmp_rs['fov_radius'].values[0], alpha=0.1)
#             ax.add_patch(p)

#         plt.xlim([-50, 50])
#         plt.ylim([-50, 50])
#         plt.legend()
#         plt.savefig('4_nodes_test/0_{opt}/overhead/{i}.png'.format(opt=opt, i=i))
#         plt.clf()


# ### PLOT METRICS

# # Plot Coverage Quality
# for opt in ['agent', 'team', 'greedy', 'random']:
#     errors = pd.read_csv('4_nodes_test/0_{opt}/surveillance_quality.csv'.format(opt=opt))
#     plt.plot(errors['time'].values, errors['value'].values, label=opt)
# plt.legend()
# plt.savefig('4_nodes_test/surveillance_quality.png')
# plt.clf()

# # Plot Errors
# for opt in ['base', 'agent', 'team', 'greedy', 'random']:
#     errors = pd.read_csv('4_nodes_test/0_{opt}/errors.csv'.format(opt=opt))
#     errors['max_error'] = errors[[str(n) for n in range(num_nodes)]].max(axis=1)
#     plt.plot(errors['time'].values, errors['max_error'].values, label=opt)
# plt.legend()
# plt.savefig('4_nodes_test/errors.png')
# plt.clf()

# # Plot Covariance
# for opt in ['base', 'agent', 'team', 'greedy', 'random']:
#     errors = pd.read_csv('4_nodes_test/0_{opt}/max_tr_cov.csv'.format(opt=opt))
#     plt.plot(errors['time'].values, errors['value'].values, label=opt)
# plt.legend()
# plt.savefig('4_nodes_test/max_tr_cov.png')
# plt.clf()

# # TODO: edge density





