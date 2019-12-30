from copy import deepcopy
import networkx as nx
import pandas as pd
import os

from ospa import *

from PHDFilterNetwork import PHDFilterNetwork
from PHDFilterNode import PHDFilterNode
from SimGenerator import SimGenerator
from target import Target


np.random.seed(42)

"""
Params
"""
fail_int = [i for i in range(50) if i in [10]]
num_nodes = 3
pos_start = np.array([-25, 0, 20])
pos_init_dist = 25
fov = 20  # radius of FOV
run_name = '3_nodes'


"""
Create Folder for Run
"""
if not os.path.exists(run_name):
    os.makedirs(run_name)
    os.makedirs(run_name + '/fail_sequence')


"""
Generate Data
"""
generator = SimGenerator(5, init_targets=[Target()])
generator.generate(50)
generator.save_data(run_name)


"""
Birth Models for entire space
"""
birthgmm = []
for x in range(-50, 50, 2):
    for y in range(-50, 50, 2):
        target = Target(init_weight=1,
                        init_state=np.array([[x], [y], [0.0], [0.0]]),
                        dt_1=0, dt_2=0)
        birthgmm.append(target)

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
saved_fail_sequence = None
for how in ['arith', 'geom']:
    for opt in ['agent', 'greedy', 'team']:
        if opt == 'team':
            mydir = 'misdp_data/inverse_covariance_matrices'
            # Clear Out Old MISDP Data
            filelist = [f for f in os.listdir(mydir) if f.endswith(".csv")]
            for f in filelist:
                os.remove(os.path.join(mydir, f))
            if os.path.exists('misdp_data/adj_mat.csv'):
                os.remove('misdp_data/adj_mat.csv')
            if os.path.exists('misdp_data/new_A.csv'):
                os.remove('misdp_data/new_A.csv')

        trial_name = run_name + '/{h}_{o}'.format(h=how, o=opt)
        print(trial_name)

        filternetwork = PHDFilterNetwork(deepcopy(node_attrs),
                                         deepcopy(weight_attrs),
                                         deepcopy(G))

        """
        Run Simulation
        """
        if how == 'arith' and opt == 'agent':
            filternetwork.step_through(generator.observations,
                                       generator.true_positions,
                                       how=how,
                                       opt=opt,
                                       fail_int=fail_int)

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
                                       fail_int=saved_fail_sequence)

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
