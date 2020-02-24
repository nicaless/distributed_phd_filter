import argparse
from copy import deepcopy
import networkx as nx
import pandas as pd
import os

from ospa import *

from PHDFilterNetwork import PHDFilterNetwork
from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target

"""
Params
"""
parser = argparse.ArgumentParser()
parser.add_argument('num', type=int, default=3)
parser.add_argument('run_name', default='3_nodes')
parser.add_argument('seed', type=int, default=42)
parser.add_argument('--single_node_fail', help='Only one node will experience failure', action='store_true')
args = parser.parse_args()

num_nodes = args.num
run_name = args.run_name
random_seed = args.seed
single_node_fail = args.single_node_fail

np.random.seed(random_seed)

total_time_steps = 50
region = [(-50, 50), (-50, 50)]  # simulation space
if single_node_fail:
    fails_before_saturation = num_nodes
else:
    fails_before_saturation = num_nodes * (num_nodes - 1) / 2 - (num_nodes - 1)
fail_freq = int(np.ceil(total_time_steps / fails_before_saturation))
# fail_int = [5, 10, 15, 20, 25, 30, 35, 40, 45]  # time steps at which failure occurs
fail_int = list(range(1, total_time_steps, fail_freq))  # time steps at which failure occurs (no failure on first time step)
x_start = -50 + (100.0 / (num_nodes + 1))  # init x coord of first node
pos_start = np.array([x_start, 0, 20])  # init x coord for all nodes
pos_init_dist = np.floor(100.0 / (num_nodes + 1))  # init x dist between nodes
fov = 20  # radius of FOV
noise_mult = [3, 3, 3, 3, 3]  # multiplier for added noise at each failure



"""
Create Folder for Run
"""
if not os.path.exists(run_name):
    os.makedirs(run_name)
    os.makedirs(run_name + '/fail_sequence')


"""
Generate Data
"""
generator = SimGenerator(init_targets=[Target()])
generator.generate(total_time_steps)
generator.save_data(run_name)


"""
Birth Models for entire space 
"""
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


"""
Create Nodes
"""
node_attrs = {}

for n in range(num_nodes):
    pos = pos_start + np.array([n*pos_init_dist, 0, 0])
    region = [(pos[0] - fov, pos[0] + fov),
              (pos[1] - fov, pos[1] + fov)]
    node_attrs[n] = PHDFilterNode(n, birthgmm,
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
count_loops = 0
saved_fail_sequence = None
for n in range(len(noise_mult)):
    noise = noise_mult[n]
    for how in ['arith', 'geom']:
        for opt in ['base', 'agent', 'greedy', 'team', 'random']:
            # if opt == 'team':
            #     mydir = 'misdp_data/inverse_covariance_matrices'
            #     # Clear Out Old MISDP Data
            #     filelist = [f for f in os.listdir(mydir) if f.endswith(".csv")]
            #     for f in filelist:
            #         os.remove(os.path.join(mydir, f))
            #     if os.path.exists('misdp_data/adj_mat.csv'):
            #         os.remove('misdp_data/adj_mat.csv')
            #     if os.path.exists('misdp_data/new_A.csv'):
            #         os.remove('misdp_data/new_A.csv')
            #     if os.path.exists('misdp_data/new_weights.csv'):
            #         os.remove('misdp_data/new_weights.csv')

            trial_name = run_name + '/{n}_{h}_{o}'.format(n=n, h=how, o=opt)
            print(trial_name)

            filternetwork = PHDFilterNetwork(deepcopy(node_attrs),
                                             deepcopy(weight_attrs),
                                             deepcopy(G))

            """
            Run Simulation
            """
            base = opt == 'base'
            if how == 'arith' and opt == 'base':
                filternetwork.step_through(generator.observations,
                                           generator.true_positions,
                                           how=how,
                                           opt=opt,
                                           fail_int=fail_int,
                                           single_node_fail=single_node_fail,
                                           base=base,
                                           noise_mult=noise)

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
                filternetwork.step_through(generator.observations,
                                           generator.true_positions,
                                           how=how,
                                           opt=opt,
                                           fail_int=saved_fail_sequence,
                                           base=base,
                                           noise_mult=noise)

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
            filternetwork.save_topologies(trial_name + '/topologies')
            count_loops += 1


expected_num_loops = len(noise_mult) * \
                     len(['arith', 'geom']) * \
                     len(['base', 'agent', 'greedy', 'team', 'random'])

assert count_loops == expected_num_loops