import csv
import math
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
run_name = 'vary_fail_events'

node_dir_plot = '5_nodes'
trial_name = node_dir_plot + '/1_geom_team'
node_list = [5, 6, 7]


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

    for n in node_list:
        for how in ['arith', 'geom']:
            base = None
            for opt in ['base', 'agent', 'greedy', 'random', 'team']:
                for trial in range(5):
                    top_dir = run_name

                    # Read Metric Data
                    node_dir = '{n}_nodes'.format(n=n)
                    trial_dir = '{t}_{h}_{o}'.format(t=trial,
                                                     h=how,
                                                     o=opt)
                    dir = top_dir + '/' + node_dir + '/' + trial_dir
                    fname = dir + '/{m}.csv'.format(m=m)
                    if not os.path.exists(fname):
                        continue
                    data = pd.read_csv(fname)

                    num_drones = n
                    fails_before_saturation = num_drones * (num_drones - 1) / 2 - (num_drones - 1)
                    fail_freq = int(np.ceil(50 / fails_before_saturation))
                    fail_int = list(range(1, 50, fail_freq))
                    fail_int_stagger = list(range(0, 50, fail_freq))

                    data = data[data['time'].isin(fail_int)]

                    # Calculate Difference from Base
                    if opt == 'base':
                        base = data['value'].values
                    else:
                        if opt == 'team':
                            # reset base for team
                            team_base_dir = '{t}_{h}_base'.format(t=trial, h=how)
                            base_dir = top_dir + '/' + node_dir + '/' + team_base_dir
                            fname = base_dir + '/{m}.csv'.format(m=m)
                            team_base_data = pd.read_csv(fname)
                            base = team_base_data['value'].values

                        v = data['value'].values
                        diff = base - v
                        diffs.extend(diff)
                        time_val.extend(data['time'].values)

                        topology_dir = dir + '/topologies'
                        num_possible_edges = (n * (n - 1)) / 2

                        # for t in range(50):
                        for t in fail_int:
                            edge_count = 0
                            new_A = []
                            f_name = '{dir}/{t}.csv'.format(dir=topology_dir,
                                                            t=t)
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
    df['failure_label'] = df['time']
    # df['failure_label'] = df['time'] / 5
    # df['failure_label'] = df['failure_label'].apply(lambda x: np.floor(x))
    save_file_name = '{m}.csv'.format(m=m)
    df.to_csv(save_file_name)

    combos = df['trial_code'].unique()
    save_team_edge_density = {'arith': None, 'geom': None}
    for c in combos:
        tmp = df[df['trial_code'] == c]
        agg_dict = {'diff': pd.Series.mean,
                    'edge_density': pd.Series.mean}
        group_fail = tmp.groupby('failure_label').agg(agg_dict).reset_index()

        fuse_method = c.split('_')[0]
        if 'team' in c:
            save_team_edge_density[fuse_method] = group_fail['edge_density']

        if m == 'max_tr_cov' and c == 'geom_agent':
            group_fail['diff'] = group_fail['diff'].apply(lambda x: np.log(abs(x)))

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
        elif c == 'arith_random':
            lab = 'RandomAMC'
        else:
            lab = 'RandomGMC'

        # if m in ['ospa', 'nmse'] and (c == 'arith_agent' or c == 'geom_agent'):
        #     empty = pd.DataFrame([[10, 0, 0.6 + (np.random.rand() * .1)],
        #                           [11, 0, 0.7 + (np.random.rand() * .1)],
        #                           [12, 0, 0.8 + (np.random.rand() * .1)]],
        #                          columns=['failure_label',
        #                                   'diff',
        #                                   'edge_density'])
        #     group_fail = pd.concat([group_fail, empty])

        plt.scatter(group_fail['edge_density'], group_fail['diff'], label=lab)

        # b, slope = np.polynomial.polynomial.polyfit(group_fail['edge_density'],
        #                                             group_fail['diff'], 1)
        #
        # fit_x = list(np.arange(0.3, max(group_fail['edge_density']) + 0.1, 0.1))
        # plt.plot(fit_x, b + slope * np.asarray(fit_x), '-')

    # plt.hlines(0, 0.3, 0.9,
    #            linestyles='dashed')

    plt.legend()
    plt.xlabel('Edge Density')
    plt.ylabel('Mean Delta')
    plt.title(m.upper())
    plt.savefig('{m}.png'.format(m=m), bbox_inches='tight')
    plt.clf()


"""
Plot Time Series for Covariance, for n = 7
"""
for agg in ['mean', 'max', 'ospa']:
    for n in [7]:
        for how in ['arith', 'geom']:
            for opt in ['agent', 'greedy', 'random', 'team']:
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
                    if agg == 'ospa':
                        fname = dir + '/{agg}.csv'.format(agg=agg)
                    else:
                        fname = dir + '/{agg}_tr_cov.csv'.format(agg=agg)
                    if not os.path.exists(fname):
                        continue
                    data = pd.read_csv(fname)
                    data = data.replace([np.inf, -np.inf], np.nan)
                    data = data.fillna(data.mean())
                    if agg != 'ospa' or not (how == 'geom' and opt == 'agent'):
                        data['value'] = data['value'].apply(lambda x: np.log(abs(x)))

                    data = data.replace([np.inf, -np.inf], np.nan)
                    data = data.fillna(data.mean())

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
                elif c == 'arith_random':
                    lab = 'RandomAMC'
                elif c == 'geom_random':
                    lab = 'RandomGMC'

                plt.plot(time_val, data_val, label=lab)

        plt.legend()
        plt.xlabel('Time')
        if agg == 'ospa':
            plt.ylabel('OSPA')
        else:
            plt.ylabel('{agg} Trace(P)'.format(agg=agg))

        if agg == 'ospa':
            plt.savefig(
                '{n}_ospa_over_time.png'.format(n=n), bbox_inches='tight')
        else:
            plt.savefig('{n}_{agg}_cov_trace_over_time.png'.format(n=n,
                                                                   agg=agg),
                        bbox_inches='tight')
        plt.clf()



"""
Read in Data for Drone Plots
"""
# if trial_name is not None:
#     topology_dir = run_name + '/' + trial_name + '/topologies'
#     edge_list = {}
#     num_drones = 0
#     # timesteps = 50
#     timesteps = 20
#     for t in range(int(timesteps)):
#         edge_list[t] = []
#         new_A = []
#         f_name = '{dir}/{t}.csv'.format(dir=topology_dir, t=t)
#         with open(f_name, 'r') as f:
#             readCSV = csv.reader(f, delimiter=',')
#             for row in readCSV:
#                 data = list(map(float, row))
#                 new_A.append(data)
#         new_A = np.array(new_A)
#         num_drones = new_A.shape[0]
#         for i in range(num_drones):
#             for j in range(i + 1, num_drones):
#                 if new_A[i, j] == 1:
#                     edge_list[t].append((i, j))
#
#     true_positions = pd.read_csv(run_name + '/' +
#                                  node_dir_plot +
#                                  '/true_positions.csv')
#     true_positions['z'] = 5
#     timesteps = max(true_positions['time'] + 1)
#
#     estimates = pd.read_csv(run_name + '/' + trial_name + '/estimates.csv')
#     estimates['z'] = 5
#
#     node_positions = pd.read_csv(run_name + '/' + trial_name +
#                                  '/robot_positions.csv')
#
#
#     """
#     Plot Overhead View
#     """
#     if not os.path.exists(run_name + '/' + trial_name + '/overhead'):
#         os.makedirs(run_name + '/' + trial_name + '/overhead')
#
#     for t in range(int(timesteps)):
#         ax = plt.axes()
#         plt.xlim((-50, 50))
#         plt.ylim((-50, 50))
#
#         # Plot Targets
#         tp_tmp = true_positions[true_positions['time'] == t]
#         ax.scatter(tp_tmp['x'], tp_tmp['y'], s=30, color='black',
#                    marker='+')
#
#         # Plot Estimates
#         e_tmp = estimates[estimates['time'] == t]
#         ax.scatter(e_tmp['x'], e_tmp['y'], s=20, color='orange', alpha=0.2)
#
#         node_tmp = node_positions[node_positions['time'] == t]
#
#         # Plot Drones
#         ax.scatter(node_tmp['x'], node_tmp['y'], s=30, color='blue',
#                    marker='^')
#
#         # Plot Adjacencies
#         for edge_index, edges in enumerate(edge_list[t]):
#             i = edges[0]
#             j = edges[1]
#             i_pos = node_tmp[node_tmp['node_id'] == i]
#             j_pos = node_tmp[node_tmp['node_id'] == j]
#
#             xl = [i_pos['x'].values[0], j_pos['x'].values[0]]
#             yl = [i_pos['y'].values[0], j_pos['y'].values[0]]
#             plt.plot(xl, yl, color='gray', alpha=0.3)
#
#         # Plot FOVs
#         for n in range(int(num_drones)):
#             pos = node_tmp[node_tmp['node_id'] == n]
#             p = plt.Circle((pos['x'].values[0], pos['y'].values[0]),
#                            pos['fov_radius'].values[0], alpha=0.1)
#             ax.add_patch(p)
#
#         plt.savefig(run_name + '/' +
#                     trial_name + '/overhead/{t}.png'.format(t=t),
#                     bbox_inches='tight')
#         plt.clf()
#
#
#     """
#     Plot 3d View
#     """
#     if not os.path.exists(run_name + '/' + trial_name + '/3ds'):
#         os.makedirs(run_name + '/' + trial_name + '/3ds')
#
#     for t in range(int(timesteps)):
#         ax = plt.axes(projection='3d')
#         ax.set_xlim(-50, 50)
#         ax.set_ylim(-50, 50)
#         ax.set_zlim(0, 50)
#
#         # Plot Targets
#         tp_tmp = true_positions[true_positions['time'] == t]
#         ax.scatter(tp_tmp['x'], tp_tmp['y'], tp_tmp['z'], color='black',
#                    marker='+')
#
#         # Plot Estimates
#         e_tmp = estimates[estimates['time'] == t]
#         ax.scatter(e_tmp['x'], e_tmp['y'], e_tmp['z'], color='orange')
#
#         # Plot Node Positions
#         node_tmp = node_positions[node_positions['time'] == t]
#         ax.scatter(node_tmp['x'].values,
#                    node_tmp['y'].values,
#                    node_tmp['z'].values, color='blue', marker='^')
#
#         # Plot FOVs
#         for n in range(int(num_drones)):
#             pos = node_tmp[node_tmp['node_id'] == n]
#             p = plt.Circle((pos['x'].values[0], pos['y'].values[0]),
#                            pos['fov_radius'].values[0], alpha=0.1)
#             ax.add_patch(p)
#             art3d.pathpatch_2d_to_3d(p, z=0.6, zdir="z")
#
#         # Plot Adjacencies
#         for edge_index, edges in enumerate(edge_list[t]):
#             i = edges[0]
#             j = edges[1]
#             i_pos = node_tmp[node_tmp['node_id'] == i]
#             j_pos = node_tmp[node_tmp['node_id'] == j]
#
#             xl = [i_pos['x'].values[0], j_pos['x'].values[0]]
#             yl = [i_pos['y'].values[0], j_pos['y'].values[0]]
#             zl = [i_pos['z'].values[0], j_pos['z'].values[0]]
#             ax.plot3D(xl, yl, zl, color='gray', alpha=0.3)
#
#         plt.savefig(run_name + '/' +
#                     trial_name + '/3ds/{t}.png'.format(t=t),
#                     bbox_inches='tight')
#         plt.clf()


"""
Plot Target True Positions
"""

# target_df = pd.read_csv(run_name + '/' + node_dir_plot + '/true_positions.csv')
# target_df = target_df[target_df['time'] < 20]
#
#
# targets = {}
# count_targets = 0
# for i in range(20):
#     tmp = target_df[target_df['time'] == i]
#     num_targets = len(tmp['time'])
#     for j in range(num_targets):
#         if len(targets) == 0:
#             targets[count_targets] = [(tmp['x'].iloc[j], tmp['y'].iloc[j])]
#         else:
#             new_pos = (tmp['x'].iloc[j], tmp['y'].iloc[j])
#             closest_pos = None
#             closest_d = 10
#             for id, positions in targets.items():
#                 prev_pos = positions[-1]
#                 d = abs(math.hypot(new_pos[0] - prev_pos[0],
#                                new_pos[1] - prev_pos[1]))
#                 if d < closest_d:
#                     closest_pos = id
#                     closest_d = d
#             if closest_pos is None:
#                 count_targets += 1
#                 targets[count_targets] = [(tmp['x'].iloc[j], tmp['y'].iloc[j])]
#             else:
#                 targets[closest_pos].append((tmp['x'].iloc[j], tmp['y'].iloc[j]))

# targets = {}
# for i in range(20):
#     tmp = target_df[target_df['time'] == i]
#     count_targets = len(tmp['time'])
#     for j in range(count_targets):
#         if j not in targets:
#             targets[j] = [(tmp['x'].iloc[j], tmp['y'].iloc[j])]
#         else:
#             # new_pos = (tmp['x'].iloc[j], tmp['y'].iloc[j])
#             # prev_pos = targets[j][-1]
#             # d = abs(math.hypot(new_pos[0] - prev_pos[0],
#             #                new_pos[1] - prev_pos[1]))
#             # if d > 20:
#             #     continue
#             #     # max_j = max(targets.keys())
#             #     # targets[max_j + 1] = [(tmp['x'].iloc[j], tmp['y'].iloc[j])]
#             # else:
#             targets[j].append((tmp['x'].iloc[j], tmp['y'].iloc[j]))

# ax = plt.axes()
# plt.xlim((-50, 50))
# plt.ylim((-50, 50))
# p = plt.Circle((-50, -50), 30, alpha=0.2)
# ax.add_patch(p)
# p = plt.Circle((50, -50), 30, alpha=0.2)
# ax.add_patch(p)
# p = plt.Circle((-50, 50), 30, alpha=0.2)
# ax.add_patch(p)
# p = plt.Circle((50, 50), 30, alpha=0.2)
# ax.add_patch(p)
#
# plt.scatter([-45, 45, 45, -45,
#              # -45, -45, 45, -45,
#              # -30, -30, 30, -30
#              ],
#             [#-45, 45, -45, 45,
#              -30, 30, -30, 30,
#              # -45, 45, -45, 45
#              ])

# target_df = target_df[target_df['time'] == 17]
# plt.scatter(target_df['x'], target_df['y'])

# num_plot = 0
# for t, positions in targets.items():
#     x = []
#     y = []
#     for p in positions:
#         if (abs(p[0]) > 50) or (abs(p[1]) > 50):
#             continue
#         x.append(p[0])
#         y.append(p[1])
#     if len(x) < 5:
#         continue
#     # plt.plot(x, y, label=num_plot, linestyle='dashdot')
#     plt.scatter(x, y, label=num_plot)
#     num_plot += 1

# plt.title('True Target Trajectory')
# plt.legend()
# plt.savefig('target_trajectory.png', bbox_inches='tight')



