import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
params = {
    'axes.labelsize': 16,
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'text.usetex': False,
    'figure.figsize': [8, 4]  # instead of 4.5, 4.5
}
mpl.rcParams.update(params)


dir = 'tro_redo/7_nodes_4_targets'
dir_team = 'tro_redo_team/7_nodes_4_targets'
known_dir = 'tro_redo/7_nodes_4_targets_known_input'
# known_dir_team =
num_nodes = 7

total_time_steps = 50
fails_before_saturation = num_nodes * (num_nodes - 1) / 2 - (num_nodes - 1)
fail_freq = int(np.ceil(total_time_steps / fails_before_saturation))
fail_int = list(range(1, total_time_steps, fail_freq))

# Plot Coverage Quality
plt.figure(figsize=[8,4])
ax = plt.axes()
num_fail = 1
xticks = []
means = {'agent': [], 'team': [], 'greedy': []}
for f in range(1, len(fail_int)):
    all_errors = {}
    base = None
    team_base = None
    for i in range(10):
        for opt in ['base', 'agent', 'team', 'greedy']:
            # if opt == 'base':
            #     errors = pd.read_csv(dir + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))

            errors['value'] = errors['value'].abs()
            # errors['value'] = errors['value'].apply(lambda x: 2 if abs(x) > 2 else abs(x))
            errors['value'] = errors['value'].apply(lambda x: np.log(x))

            # errors['value'] = np.log(errors['value'])
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            if opt == 'base':
                base = np.array(errors['value'].values)

                team_errors = pd.read_csv(dir_team + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))

                team_errors['value'] = team_errors['value'].abs()
                team_errors['value'] = team_errors['value'].apply(lambda x: np.log(x))
                # team_errors['value'] = team_errors['value'].apply(lambda x: 2 if abs(x) > 2 else abs(x))

                team_errors = team_errors[
                    (team_errors['time'] >= fail_int[f] - fail_freq) & (team_errors['time'] < fail_int[f])]
                team_base = np.array(team_errors['value'].values)
                continue
            else:
                if opt == 'team':
                    errors = (team_base - np.array(errors['value'].values))
                else:
                    errors = (base - np.array(errors['value'].values))

            if opt not in all_errors.keys():
                # all_errors[opt] = list(errors['value'].values)
                all_errors[opt] = list(errors)
            else:
                # all_errors[opt].extend(errors['value'].values)
                all_errors[opt].extend(errors)

    data = [all_errors['agent'], all_errors['team'], all_errors['greedy']]
    means['agent'].append(np.mean(all_errors['agent']))
    means['team'].append(np.mean(all_errors['team']))
    means['greedy'].append(np.mean(all_errors['greedy']))

    xticks.append(num_fail + 1)
    bp = plt.boxplot(data, positions=[num_fail, num_fail+1, num_fail+2], widths=0.6, showfliers=False)
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][1], color='orange')
    plt.setp(bp['boxes'][2], color='green')

    plt.setp(bp['medians'][0], color='blue')
    plt.setp(bp['medians'][1], color='orange')
    plt.setp(bp['medians'][2], color='green')

    # plt.setp(bp['fliers'][0], color='blue')
    # plt.setp(bp['fliers'][1], color='orange')
    # plt.setp(bp['fliers'][2], color='green')
    num_fail += 4

hB, = plt.plot([1,1],'blue')
hO, = plt.plot([1,1],'orange')
hG, = plt.plot([1,1],'green')
plt.legend((hB, hO, hG), ('ACCG', 'TCCG', 'greedy'), frameon=True, loc=(1.04,0))

hB.set_visible(False)
hO.set_visible(False)
hG.set_visible(False)

# ax.set_yscale('log')
ax.set_xticklabels(list(range(1, len(fail_int))))
ax.set_xticks(xticks)
# plt.ylabel('Distribution of Estimation Errors  \n (Improvement over baseline)')
plt.ylabel('Delta of Coverage Performance')
plt.xlabel('Event')

plt.savefig(dir + '/surveillance_quality_bp.png', bbox_inches='tight')
plt.clf()

plt.plot(list(range(1, len(fail_int))), means['agent'], label='ACCG')
plt.plot(list(range(1, len(fail_int))), means['team'], label='TCCG')
plt.plot(list(range(1, len(fail_int))), means['greedy'], label='greedy')
plt.legend(frameon=True, loc=(1.04,0))
# plt.ylabel('Average Estimation Errors  \n (Improvement over baseline)')
plt.ylabel('Average Delta of Coverage Performance')
plt.xlabel('Event')
plt.savefig(dir + '/surveillance_quality_avg.png', bbox_inches='tight')
plt.clf()

unknown_surv_errors = means



# Plot Errors
plt.figure(figsize=[8,4])
ax = plt.axes()
num_fail = 1
xticks = []
means = {'agent': [], 'team': [], 'greedy': []}
for f in range(1, len(fail_int)):
    all_errors = {}
    base = None
    team_base = None
    for i in range(10):
        for opt in ['base', 'agent', 'team', 'greedy']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))

            for n in range(num_nodes):
                errors[str(n)] = errors[str(n)].abs()
                # errors[str(n)] = errors[str(n)].apply(lambda x: 2 if abs(x) > 2 else abs(x))
                errors[str(n)] = errors[str(n)].apply(lambda x: np.log(x))

            errors['value'] = errors[[str(n) for n in range(num_nodes)]].mean(axis=1)
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            if opt == 'base':
                base = np.array(errors['value'].values)

                team_errors = pd.read_csv(dir_team + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))

                for n in range(num_nodes):
                    team_errors[str(n)] = team_errors[str(n)].abs()
                    # team_errors[str(n)] = team_errors[str(n)].apply(lambda x: 2 if abs(x) > 2 else abs(x))
                    team_errors[str(n)] = team_errors[str(n)].apply(lambda x: np.log(x))

                team_errors['value'] = team_errors[[str(n) for n in range(num_nodes)]].mean(axis=1)
                team_errors = team_errors[(team_errors['time'] >= fail_int[f] - fail_freq) & (team_errors['time'] < fail_int[f])]
                team_base = np.array(team_errors['value'].values)
                continue
            else:
                if opt == 'team':
                    errors = (team_base - np.array(errors['value'].values))
                else:
                    errors = (base - np.array(errors['value'].values))

            if opt not in all_errors.keys():
                # all_errors[opt] = list(errors['value'].values)
                all_errors[opt] = list(errors)
            else:
                # all_errors[opt].extend(errors['value'].values)
                all_errors[opt].extend(errors)

    data = [all_errors['agent'], all_errors['team'], all_errors['greedy']]
    means['agent'].append(np.mean(all_errors['agent']))
    means['team'].append(np.mean(all_errors['team']))
    means['greedy'].append(np.mean(all_errors['greedy']))

    xticks.append(num_fail + 1)
    bp = plt.boxplot(data, positions=[num_fail, num_fail+1, num_fail+2], widths=0.6, showfliers=False)
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][1], color='orange')
    plt.setp(bp['boxes'][2], color='green')

    plt.setp(bp['medians'][0], color='blue')
    plt.setp(bp['medians'][1], color='orange')
    plt.setp(bp['medians'][2], color='green')

    # plt.setp(bp['fliers'][0], color='blue')
    # plt.setp(bp['fliers'][1], color='orange')
    # plt.setp(bp['fliers'][2], color='green')
    num_fail += 4

hB, = plt.plot([1,1],'blue')
hO, = plt.plot([1,1],'orange')
hG, = plt.plot([1,1],'green')
plt.legend((hB, hO, hG), ('ACCG', 'TCCG', 'greedy'), frameon=True, loc=(1.04,0))

hB.set_visible(False)
hO.set_visible(False)
hG.set_visible(False)

# ax.set_yscale('log')
ax.set_xticklabels(list(range(1, len(fail_int))))
ax.set_xticks(xticks)
# plt.ylabel('Distribution of Estimation Errors  \n (Improvement over baseline)')
plt.ylabel('Delta of Est. Errors')
plt.xlabel('Event')

plt.savefig(dir + '/errors_bp.png', bbox_inches='tight')
plt.clf()

plt.plot(list(range(1, len(fail_int))), means['agent'], label='ACCG')
plt.plot(list(range(1, len(fail_int))), means['team'], label='TCCG')
plt.plot(list(range(1, len(fail_int))), means['greedy'], label='greedy')
plt.legend(frameon=True, loc=(1.04,0))
# plt.ylabel('Average Estimation Errors  \n (Improvement over baseline)')
plt.ylabel('Average Delta of Est. Errors')
plt.xlabel('Event')
plt.savefig(dir + '/errors_avg.png', bbox_inches='tight')
plt.clf()

unknown_input_errors = means




# Plot Covariance
ax = plt.axes()
num_fail = 1
xticks = []
means = {'agent': [], 'team': [], 'greedy': []}
for f in range(1, len(fail_int)):
    all_errors = {}
    base = None
    team_base = None
    for i in range(10):
        for opt in ['base', 'agent', 'team', 'greedy']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))

            errors['value'] = errors['value'].abs()
            # errors['value'] = errors['value'].apply(lambda x: 2 if abs(x) > 2 else abs(x))
            errors['value'] = errors['value'].apply(lambda x: np.log(x))

            # errors['value'] = np.log(errors['value'])
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            if opt == 'base':
                base = np.array(errors['value'].values)

                team_errors = pd.read_csv(dir_team + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))

                team_errors['value'] = team_errors['value'].abs()
                team_errors['value'] = team_errors['value'].apply(lambda x: np.log(x))
                # team_errors['value'] = team_errors['value'].apply(lambda x: 2 if abs(x) > 2 else abs(x))

                team_errors = team_errors[
                    (team_errors['time'] >= fail_int[f] - fail_freq) & (team_errors['time'] < fail_int[f])]
                team_base = np.array(team_errors['value'].values)
                continue
            else:
                if opt == 'team':
                    errors = (team_base - np.array(errors['value'].values))
                else:
                    errors = (base - np.array(errors['value'].values))

            if opt not in all_errors.keys():
                # all_errors[opt] = list(errors['value'].values)
                all_errors[opt] = list(errors)
            else:
                # all_errors[opt].extend(errors['value'].values)
                all_errors[opt].extend(errors)

    data = [all_errors['agent'], all_errors['team'], all_errors['greedy']]
    means['agent'].append(np.mean(all_errors['agent']))
    means['team'].append(np.mean(all_errors['team']))
    means['greedy'].append(np.mean(all_errors['greedy']))

    xticks.append(num_fail + 1)
    bp = plt.boxplot(data, positions=[num_fail, num_fail+1, num_fail+2], widths=0.6, showfliers=False)
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][1], color='orange')
    plt.setp(bp['boxes'][2], color='green')

    plt.setp(bp['medians'][0], color='blue')
    plt.setp(bp['medians'][1], color='orange')
    plt.setp(bp['medians'][2], color='green')

    # plt.setp(bp['fliers'][0], color='blue')
    # plt.setp(bp['fliers'][1], color='orange')
    # plt.setp(bp['fliers'][2], color='green')
    num_fail += 4

hB, = plt.plot([1,1],'blue')
hO, = plt.plot([1,1],'orange')
hG, = plt.plot([1,1],'green')
plt.legend((hB, hO, hG), ('ACCG', 'TCCG', 'greedy'), frameon=True, loc=(1.04,0))
hB.set_visible(False)
hO.set_visible(False)
hG.set_visible(False)

# ax.set_yscale('log')

ax.set_xticklabels(list(range(1, len(fail_int))))
ax.set_xticks(xticks)

plt.ylabel('Delta of Max Trace(P)')
plt.xlabel('Event')

plt.savefig(dir + '/max_tr_cov_bp.png', bbox_inches='tight')
plt.clf()

plt.plot(list(range(1, len(fail_int))), means['agent'], label='ACCG')
plt.plot(list(range(1, len(fail_int))), means['team'], label='TCCG')
plt.plot(list(range(1, len(fail_int))), means['greedy'], label='greedy')
plt.legend(frameon=True, loc=(1.04,0))
# plt.ylabel('Average Max Trace of Covariance  \n (Improvement over baseline)')
plt.ylabel('Average Delta of Max Trace(P)')
plt.xlabel('Event')
plt.savefig(dir + '/max_tr_cov_avg.png', bbox_inches='tight')
plt.clf()

unknown_input_cov = means



# params = {
#     'axes.labelsize': 12,
#     'font.size': 12,
#     'font.family': 'sans-serif',
#     'font.sans-serif': 'Helvetica',
#     'text.usetex': False,
#     'figure.figsize': [5, 5]  # instead of 4.5, 4.5
# }
# mpl.rcParams.update(params)
# # Plot Targets
# # for opt in ['agent', 'team', 'greedy', 'random']:
# for opt in ['agent', 'team', 'greedy']:
#     if opt == 'team':
#         est = pd.read_csv(dir_team + '/0_{opt}/estimates.csv'.format(opt=opt))
#         robot_pos = pd.read_csv(dir_team + '/0_{opt}/robot_positions.csv'.format(opt=opt))
#     else:
#         est = pd.read_csv(dir + '/0_{opt}/estimates.csv'.format(opt=opt))
#         robot_pos = pd.read_csv(dir + '/0_{opt}/robot_positions.csv'.format(opt=opt))
#     colors = ['red', 'blue', 'green', 'orange']
#     tops = [0] + fail_int
#     for i in tops:
#         ax = plt.axes()
#         rs = robot_pos[robot_pos['time'] == i]
#         for t in range(4):
#             if opt == 'team':
#                 df = pd.read_csv(dir_team + '/0_{opt}/target_{t}_positions.csv'.format(opt=opt, t=t))
#             else:
#                 df = pd.read_csv(dir + '/0_{opt}/target_{t}_positions.csv'.format(opt=opt, t=t))
#             tmp = df.loc[i - 5 + 1: i + 2]
#
#             # plt.plot(tmp['x'].values, tmp['y'].values, '+',
#             #          label="True Target {t}".format(t=t), alpha=0.8, color=colors[t], s=30)
#             ax.scatter(tmp['x'].values, tmp['y'].values,
#                        label="True Target {t}".format(t=t), alpha=0.8, color=colors[t], s=35)
#
#             e = est[est['target'] == t]
#             e = e.groupby('time').agg({'x': 'mean', 'y': 'mean'}).reset_index()
#             e = e[(e['time'] >= i - 5) & (e['time'] <= i+1)]
#             plt.plot(e['x'].values, e['y'].values, '--',
#                      label="Estimate {t}".format(t=t), alpha=0.5, color=colors[t])
#
#         plt.scatter(rs['x'].values, rs['y'].values, color='black', marker='x', s=40)
#
#         # Plot Adjacencies
#         edge_list = []
#         new_A = []
#         if opt == 'team':
#             topology_file = dir_team + '/0_{opt}/topologies/{i}.csv'.format(opt=opt, i=i)
#         else:
#             topology_file = dir + '/0_{opt}/topologies/{i}.csv'.format(opt=opt, i=i)
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
#
#                     xl = [n_pos['x'].values[0], o_pos['x'].values[0]]
#                     yl = [n_pos['y'].values[0], o_pos['y'].values[0]]
#                     plt.plot(xl, yl, color='gray', alpha=0.5)
#
#             # Plot FOV
#             tmp_rs = rs[rs['node_id'] == n]
#             p = plt.Circle((tmp_rs['x'].values[0], tmp_rs['y'].values[0]), tmp_rs['fov_radius'].values[0], alpha=0.1)
#             ax.add_patch(p)
#
#         plt.xlim([-50, 50])
#         plt.ylim([-50, 50])
#         # plt.legend()
#         plt.savefig(dir + '/0_{opt}/overhead/{i}.png'.format(opt=opt, i=i))
#         plt.clf()
#
#
# # Plot Overhead of video
#
# colors = ['red', 'blue', 'green', 'orange']
# tops = [0, 5, 10, 15, 20]
# fail_nodes = [1, 2, 3, 4]
# fnode = [-1]
# robot_pos = pd.read_csv('crazyflie/robot_positions.csv')
# for i in range(25):
# # for i in tops:
#     ax = plt.axes()
#     rs = robot_pos[robot_pos['time'] == i]
#     for t in range(4):
#         df = pd.read_csv('crazyflie/target_{t}_positions.csv'.format(t=t))
#         tmp = df.loc[i]
#         ax.scatter(tmp['x'], tmp['y'],
#                    label="True Target {t}".format(t=t), alpha=0.8, color='teal', s=35)
#
#     user = pd.read_csv('crazyflie/user_positions.csv')
#     plt.scatter(user['0'].iloc[i // 5], user['1'].iloc[i // 5], color='blue', marker="^", s=40)
#
#     plt.scatter(rs['x'].values, rs['y'].values, color='black', marker='x', s=40)
#
#     # Plot
#     if i in [5, 10, 15, 20]:
#         fnode.append(fail_nodes[(i // 5) - 1])
#
#     rs_filter = rs[rs['node_id'].isin(fnode)]
#     if len(rs_filter) > 0:
#         plt.scatter(rs_filter['x'].values, rs_filter['y'].values, color='red', marker='x', s=40)
#
#     # Plot Adjacencies
#     edge_list = []
#     new_A = []
#     topology_file = 'crazyflie/topologies/{i}.csv'.format(i=i)
#     with open(topology_file, 'r') as f:
#         readCSV = csv.reader(f, delimiter=',')
#         for row in readCSV:
#             data = list(map(float, row))
#             new_A.append(data)
#     new_A = np.array(new_A)
#     num_drones = new_A.shape[0]
#     for n in range(num_drones):
#         for o in range(n+1, num_drones):
#             if new_A[n, o] == 1:
#                 n_pos = rs[rs['node_id'] == n]
#                 o_pos = rs[rs['node_id'] == o]
#
#                 xl = [n_pos['x'].values[0], o_pos['x'].values[0]]
#                 yl = [n_pos['y'].values[0], o_pos['y'].values[0]]
#                 plt.plot(xl, yl, color='gray', alpha=0.5)
#
#         # Plot FOV
#         tmp_rs = rs[rs['node_id'] == n]
#         p = plt.Circle((tmp_rs['x'].values[0], tmp_rs['y'].values[0]), tmp_rs['fov_radius'].values[0], alpha=0.1)
#         ax.add_patch(p)
#
#     plt.xlim([-10, 10])
#     plt.ylim([-10, 10])
#     # plt.legend()
#     plt.savefig('crazyflie/overhead/{i}.png'.format(i=i))
#     plt.clf()


# Compare With Input and Without Input

params = {
    'axes.labelsize': 16,
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'text.usetex': False,
    'figure.figsize': [8, 4]  # instead of 4.5, 4.5
}
mpl.rcParams.update(params)
errors_known = {'agent': [], 'team': []}
errors_unknown = {'agent': [], 'team': []}
for f in range(1, len(fail_int)):
    errors_agg = {'agent': [], 'team': []}
    for i in range(10):
        for opt in ['agent', 'team']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(dir_team + '/fail_sequence/_node_list.csv'.format(i=i, opt=opt),header=None)
            else:
                errors = pd.read_csv(known_dir + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(known_dir + '/fail_sequence/_node_list.csv'.format(i=i,opt=opt), header=None)

            for n in range(num_nodes):
                errors[str(n)] = errors[str(n)].abs()
                errors[str(n)] = errors[str(n)].apply(lambda x: np.log(x))

            # errors['value'] = errors[[str(n) for n in range(num_nodes)]].mean(axis=1)
            fail_node = fail_seq[fail_seq[0] == fail_int[f]][1].values[0]
            errors['value'] = errors[str(fail_node)]
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            errors_agg[opt].extend(errors['value'].values)

    errors_known['agent'].append(np.mean(errors_agg['agent']))
    errors_known['team'].append(np.mean(errors_agg['team']))

    errors_agg = {'agent': [], 'team': []}
    for i in range(10):
        for opt in ['agent', 'team']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(dir_team + '/fail_sequence/_node_list.csv'.format(i=i, opt=opt), header=None)
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(dir + '/fail_sequence/_node_list.csv'.format(i=i, opt=opt), header=None)

            for n in range(num_nodes):
                errors[str(n)] = errors[str(n)].abs()
                errors[str(n)] = errors[str(n)].apply(lambda x: np.log(x))

            # errors['value'] = errors[[str(n) for n in range(num_nodes)]].mean(axis=1)
            fail_node = fail_seq[fail_seq[0] == fail_int[f]][1].values[0]
            errors['value'] = errors[str(fail_node)]
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            errors_agg[opt].extend(errors['value'].values)

    errors_unknown['agent'].append(np.mean(errors_agg['agent']))
    errors_unknown['team'].append(np.mean(errors_agg['team']))

plt.plot(list(range(1, len(fail_int))), np.abs(errors_unknown['agent']), '--', label='ACCG - Unknown Input', color='blue')
plt.plot(list(range(1, len(fail_int))), np.abs(errors_unknown['team']), '--', label='TCCG - Unknown Input', color='orange')

plt.plot(list(range(1, len(fail_int))), np.abs(errors_known['agent']), linestyle='dotted', label='ACCGK - Known Input', color='blue')
plt.plot(list(range(1, len(fail_int))), np.abs(errors_known['team']), linestyle='dotted', label='TCCGK - Known Input', color='orange')



plt.legend(frameon=True, loc=(1.04,0))
plt.ylabel('Average Estimation Error')
plt.xlabel('Event')
# plt.show()
plt.savefig(dir + '/errors_compare_to_known_input.png', bbox_inches='tight')
plt.clf()


errors_known = {'agent': [], 'team': []}
errors_unknown = {'agent': [], 'team': []}
for f in range(1, len(fail_int)):
    errors_agg = {'agent': [], 'team': []}
    for i in range(10):
        for opt in ['agent', 'team']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/mean_tr_cov.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(dir_team + '/fail_sequence/_node_list.csv'.format(i=i, opt=opt), header=None)
            else:
                errors = pd.read_csv(known_dir + '/{i}_{opt}/mean_tr_cov.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(known_dir + '/fail_sequence/_node_list.csv'.format(i=i,opt=opt), header=None)

            errors['value'] = errors['value'].abs()
            errors['value'] = errors['value'].apply(lambda x: np.log(x))
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            errors_agg[opt].extend(errors['value'].values)

    errors_known['agent'].append(np.mean(errors_agg['agent']))
    errors_known['team'].append(np.mean(errors_agg['team']))

    errors_agg = {'agent': [], 'team': []}
    for i in range(10):
        for opt in ['agent', 'team']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/mean_tr_cov.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(dir_team + '/fail_sequence/_node_list.csv'.format(i=i, opt=opt), header=None)
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/mean_tr_cov.csv'.format(i=i, opt=opt))
                fail_seq = pd.read_csv(dir + '/fail_sequence/_node_list.csv'.format(i=i, opt=opt), header=None)

            errors['value'] = errors['value'].abs()
            errors['value'] = errors['value'].apply(lambda x: np.log(x))
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]

            errors_agg[opt].extend(errors['value'].values)

    errors_unknown['agent'].append(np.mean(errors_agg['agent']))
    errors_unknown['team'].append(np.mean(errors_agg['team']))

plt.plot(list(range(1, len(fail_int))), np.abs(errors_unknown['agent']), '--', label='ACCG - Unknown Input', color='blue')
plt.plot(list(range(1, len(fail_int))), np.abs(errors_unknown['team']), '--', label='TCCG - Unknown Input', color='orange')

plt.plot(list(range(1, len(fail_int))), np.abs(errors_known['agent']), linestyle='dotted', label='ACCGK - Known Input', color='blue')
plt.plot(list(range(1, len(fail_int))), np.abs(errors_known['team']), linestyle='dotted', label='TCCGK - Known Input', color='orange')

plt.legend(frameon=True, loc=(1.04,0))
plt.ylabel('Average Tr(P)')
plt.xlabel('Event')
# plt.show()
plt.savefig(dir + '/max_tr_cov_compare_to_known_input.png', bbox_inches='tight')
plt.clf()
