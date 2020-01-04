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
params = {
    'axes.labelsize': 20,
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'text.usetex': False,
    'figure.figsize': [7, 4]  # instead of 4.5, 4.5
}
mpl.rcParams.update(params)


"""
Params
"""
run_name = 'multi_fail_nosiman_redogeom'

node_dir_plot = '5_nodes'
trial_name = node_dir_plot + '/1_arith_agent'


"""
Plot Scatter for Errors, Covariance, OSPA, NMSE
"""
# avg metric between failures and take difference

for m in ['errors', 'max_tr_cov', 'mean_tr_cov', 'ospa', 'nmse']:
    time_val = []
    diffs = []
    time_val_edge = []
    edge_count_list = []
    edge_density = []
    trial_code = []

    # for n in [5, 6, 7, 10, 12]:
    for n in [5, 6, 7]:
        for how in ['arith', 'geom']:
            base = None
            for opt in ['base', 'agent', 'greedy', 'team']:
                if opt == 'team' and n > 7:
                    continue
                for trial in range(5):
                    if opt == 'team':
                        top_dir = run_name + '_team'
                    else:
                        top_dir = run_name

                    # Read Metric Data
                    node_dir = '{n}_nodes'.format(n=n)
                    trial_dir = '{t}_{h}_{o}'.format(t=trial,
                                                     h=how,
                                                     o=opt)
                    dir = top_dir + '/' + node_dir + '/' + trial_dir
                    fname = dir + '/{m}.csv'.format(m=m)
                    data = pd.read_csv(fname)

                    # Calculate Difference from Base
                    if opt == 'base':
                        base = data['value'].values
                    else:
                        v = data['value'].values
                        diff = base - v
                        diffs.extend(diff)
                        time_val.extend(data['time'].values)

                        topology_dir = dir + '/topologies'
                        num_drones = n
                        num_possible_edges = (n * (n -1)) / 2
                        for t in range(35):
                            edge_count = 0
                            new_A = []
                            f_name = '{dir}/{t}.csv'.format(dir=topology_dir, t=t)
                            with open(f_name, 'r') as f:
                                readCSV = csv.reader(f, delimiter=',')
                                for row in readCSV:
                                    data = list(map(float, row))
                                    new_A.append(data)
                            new_A = np.array(new_A)
                            for i in range(num_drones):
                                for j in range(i + 1, num_drones):
                                    if new_A[i, j] == 1:
                                        edge_count += 1
                            time_val_edge.append(t)
                            edge_count_list.append(edge_count)
                            edge_density.append(edge_count / float(num_possible_edges))
                            trial_code.append('{h}_{o}'.format(h=how, o=opt))

    df = pd.DataFrame([time_val, diffs, edge_density, trial_code])
    df = df.transpose()
    df.columns = ['time', 'diff', 'edge_density', 'trial_code']
    df['failure_label'] = df['time'] / 5
    df['failure_label'] = df['failure_label'].apply(lambda x: np.floor(x))
    save_file_name = '{m}.csv'.format(m=m)
    df.to_csv(save_file_name)

    combos = df['trial_code'].unique()
    for c in combos:
        tmp = df[df['trial_code'] == c]
        agg_dict = {'diff': pd.Series.mean,
                    'edge_density': pd.Series.mean}
        group_fail = tmp.groupby('failure_label').agg(agg_dict).reset_index()
        if m == 'max_tr_cov':
            group_fail['diff'] = group_fail['diff'].apply(lambda x: np.log(x))
        if c == 'arith_agent':
            lab = 'RCAMC'
        elif c == 'arith_greedy':
            lab = 'GreedyAMC'
        elif c == 'arith_team':
            lab = 'TCAMC'
        elif c == 'geom_agent':
            lab = 'RCGMC'
        elif c == 'geom_greedy':
            lab = 'GreedyGMC'
        elif c == 'geom_team':
            lab = 'TCGMC'
        plt.scatter(group_fail['edge_density'], group_fail['diff'], label=lab)

    plt.hlines(0, 0.3, max(group_fail['edge_density']) + 0.1,
               linestyles='dashed')

    plt.legend()
    plt.xlabel('Edge Density')
    plt.ylabel('Mean Delta')
    plt.title(m.upper())
    plt.savefig('{m}.png'.format(m=m), bbox_inches='tight')
    plt.clf()


"""
Plot Time Series for Covariance, for n = 7
"""
for n in [5, 6, 7]:
    for how in ['arith', 'geom']:
        for opt in ['agent', 'greedy', 'team']:
            if opt == 'team':
                top_dir = run_name + '_team'
            else:
                top_dir = run_name

            time_val = []
            data_val = []

            for trial in range(5):
                # Read Metric Data
                node_dir = '{n}_nodes'.format(n=n)
                trial_dir = '{t}_{h}_{o}'.format(t=trial,
                                                 h=how,
                                                 o=opt)
                dir = top_dir + '/' + node_dir + '/' + trial_dir
                fname = dir + '/max_tr_cov.csv'
                data = pd.read_csv(fname)
                data['value'] = data['value'].apply(lambda x: np.log(x))

                if len(data_val) == 0:
                    time_val.extend(data['time'].values)
                    data_val.extend(data['value'].values)
                else:
                    data_val = data_val + data['value'].values

            data_val = [v / 5.0 for v in data_val]

            c = '{h}_{o}'.format(h=how, o=opt)
            if c == 'arith_agent':
                lab = 'RCAMC'
            elif c == 'arith_greedy':
                lab = 'GreedyAMC'
            elif c == 'arith_team':
                lab = 'TCAMC'
            elif c == 'geom_agent':
                lab = 'RCGMC'
            elif c == 'geom_greedy':
                lab = 'GreedyGMC'
            elif c == 'geom_team':
                lab = 'TCGMC'

            plt.plot(time_val, data_val, label=lab)

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Maximum Trace(P)')
    plt.savefig('{n}_max_cov_trace_over_time.png'.format(n=n),
                bbox_inches='tight')
    plt.clf()






"""
Read in Data for Drone Plots
"""
if trial_name is not None:
    topology_dir = run_name + '/' + trial_name + '/topologies'
    edge_list = {}
    num_drones = 0
    timesteps = 35
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

    true_positions = pd.read_csv(run_name + '/' +
                                 node_dir_plot +
                                 '/true_positions.csv')
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
