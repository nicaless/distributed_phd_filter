import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

cmap = mpl.cm.get_cmap('PiYG')
normalize = mpl.colors.Normalize(vmin=0.0, vmax=100.0)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
# params = {
#     'axes.labelsize': 20,
#     'font.size': 12,
#     'font.family': 'sans-serif',
#     'font.sans-serif': 'Helvetica',
#     'text.usetex': False,
#     'figure.figsize': [7, 4]  # instead of 4.5, 4.5
# }
# mpl.rcParams.update(params)


"""
Params
"""
run_name = '3_nodes'
# trial_name = '1_arith_team'
trial_name = None


"""
Plot Errors, Covariance, OSPA, NMSE
"""
#TODO: compare against baseline combining all noise mults
# avg metric after failure (int = 10) and take difference
noise_mult = [1, 5, 10]
for noise in noise_mult:
    for m in ['errors', 'max_tr_cov', 'ospa', 'nmse']:
        base = None
        for how in ['arith', 'geom']:
            for opt in ['base', 'agent', 'greedy', 'team']:
                fname = run_name + '/{n}_{h}_{o}/{m}.csv'.format(n=noise,
                                                                 h=how, o=opt,
                                                                 m=m)
                data = pd.read_csv(fname)
                plt.plot(data['time'], data['value'],
                         label='{h}_{o}'.format(h=how, o=opt))
        plt.legend()
        plt.title(m)
        plt.savefig(run_name + '/{n}_{m}.png'.format(n=noise,
                                                     m=m), bbox_inches='tight')
        plt.clf()

topology_dir = run_name + '/' + trial_name + '/topologies'
edge_list = {}
num_drones = 0
timesteps = 20
for t in range(int(timesteps)):
    edge_list[t] = []
    new_A = []
    f_name = '{dir}/{t}.csv'.format(dir=topology_dir, t=t)
    with open(f_name, 'r') as f:
        readCSV = csv.reader(f, delimiter=',')
        for row in readCSV:
            data = list(map(float, row))
            new_A.append(data)
    new_A = np.array(new_A)
    num_drones = new_A.shape[0]
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            if new_A[i, j] == 1:
                edge_list[t].append((i, j))

"""
Read in Data for Drone Plots
"""
if trial_name is not None:
    true_positions = pd.read_csv(run_name + '/true_positions.csv')
    true_positions['z'] = 5
    timesteps = max(true_positions['time'] + 1)

    estimates = pd.read_csv(run_name + '/' + trial_name + '/estimates.csv')
    estimates['z'] = 5

    node_positions = pd.read_csv(run_name + '/' + trial_name +
                                 '/robot_positions.csv')


    """
    Plot Overhead View
    """
    if not os.path.exists(run_name + '/' + trial_name + '/overhead'):
        os.makedirs(run_name + '/' + trial_name + '/overhead')

    for t in range(int(timesteps)):
        ax = plt.axes()
        plt.xlim((-50, 50))
        plt.ylim((-50, 50))

        # Plot Targets
        tp_tmp = true_positions[true_positions['time'] == t]
        ax.scatter(tp_tmp['x'], tp_tmp['y'], tp_tmp['z'], color='black')

        # Plot Estimates
        e_tmp = estimates[estimates['time'] == t]
        ax.scatter(e_tmp['x'], e_tmp['y'], e_tmp['z'], color='orange')

        node_tmp = node_positions[node_positions['time'] == t]

        # Plot FOVs
        for n in range(int(num_drones)):
            pos = node_tmp[node_tmp['node_id'] == n]
            p = plt.Circle((pos['x'].values[0], pos['y'].values[0]),
                           pos['fov_radius'].values[0], alpha=0.1)
            ax.add_patch(p)

        plt.savefig(run_name + '/' +
                    trial_name + '/overhead/{t}.png'.format(t=t),
                    bbox_inches='tight')
        plt.clf()


    """
    Plot 3d View
    """
    if not os.path.exists(run_name + '/' + trial_name + '/3ds'):
        os.makedirs(run_name + '/' + trial_name + '/3ds')

    for t in range(int(timesteps)):
        ax = plt.axes(projection='3d')
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(0, 50)

        # Plot Targets
        tp_tmp = true_positions[true_positions['time'] == t]
        ax.scatter(tp_tmp['x'], tp_tmp['y'], tp_tmp['z'], color='black')

        # Plot Estimates
        e_tmp = estimates[estimates['time'] == t]
        ax.scatter(e_tmp['x'], e_tmp['y'], e_tmp['z'], color='orange')

        # Plot Node Positions
        node_tmp = node_positions[node_positions['time'] == t]
        ax.scatter(node_tmp['x'].values,
                   node_tmp['y'].values,
                   node_tmp['z'].values, color='blue')

        # Plot FOVs
        for n in range(int(num_drones)):
            pos = node_tmp[node_tmp['node_id'] == n]
            p = plt.Circle((pos['x'].values[0], pos['y'].values[0]),
                           pos['fov_radius'].values[0], alpha=0.1)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=0.6, zdir="z")

        # Plot Adjacencies
        for edge_index, edges in enumerate(edge_list[t]):
            i = edges[0]
            j = edges[1]
            i_pos = node_tmp[node_tmp['node_id'] == i]
            j_pos = node_tmp[node_tmp['node_id'] == j]

            xl = [i_pos['x'].values[0], j_pos['x'].values[0]]
            yl = [i_pos['y'].values[0], j_pos['y'].values[0]]
            zl = [i_pos['z'].values[0], j_pos['z'].values[0]]
            ax.plot3D(xl, yl, zl, color='gray', alpha=0.3)

        plt.savefig(run_name + '/' +
                    trial_name + '/3ds/{t}.png'.format(t=t),
                    bbox_inches='tight')
        plt.clf()
