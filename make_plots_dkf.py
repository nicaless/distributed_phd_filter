import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.art3d as art3d
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica',
    'text.usetex': False,
    'figure.figsize': [8, 4]  # instead of 4.5, 4.5
}
mpl.rcParams.update(params)


dir = '7_nodes_4_targets_trial2'
dir_team = '7_nodes_4_targets_team_only_trial2'
num_nodes = 7

total_time_steps = 50
fails_before_saturation = num_nodes * (num_nodes - 1) / 2 - (num_nodes - 1)
fail_freq = int(np.ceil(total_time_steps / fails_before_saturation))
fail_int = list(range(1, total_time_steps, fail_freq))

# Plot Coverage Quality
# for opt in ['agent', 'team', 'greedy', 'random']:
for opt in ['agent', 'team', 'greedy']:
    time = []
    all_errors = []
    for i in range(5):
        if opt == 'team':
            errors = pd.read_csv(dir_team + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))
        else:
            errors = pd.read_csv(dir + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))
        time.extend(errors['time'].values)
        all_errors.extend(errors['value'].values)
    df = pd.DataFrame([time, all_errors])
    df = df.transpose()
    df.columns = ['time', 'errors']
    df_group = df.groupby('time').agg({'errors': 'mean'}).reset_index()

    df_group = df_group[df_group['time'] > 30]

    plt.plot(df_group['time'], np.log(df_group['errors']), label=opt)
plt.legend()
plt.savefig(dir + '/surveillance_quality.png', bbox_inches='tight')
plt.clf()


ax = plt.axes()
num_fail = 1
xticks = []
for f in range(1, len(fail_int)):
    all_errors = {}
    for i in range(5):
        for opt in ['agent', 'team', 'greedy']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/surveillance_quality.csv'.format(i=i, opt=opt))

            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f] )]
            if opt not in all_errors.keys():
                all_errors[opt] = list(errors['value'].values)
            else:
                all_errors[opt].extend(errors['value'].values)

    data = [all_errors['agent'], all_errors['team'], all_errors['greedy']]

    xticks.append(num_fail+1)
    bp = plt.boxplot(data, positions=[num_fail, num_fail+1, num_fail+2], widths=0.6)
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][1], color='orange')
    plt.setp(bp['boxes'][2], color='green')

    plt.setp(bp['medians'][0], color='blue')
    plt.setp(bp['medians'][1], color='orange')
    plt.setp(bp['medians'][2], color='green')
    num_fail += 4

hB, = plt.plot([1,1],'blue')
hO, = plt.plot([1,1],'orange')
hG, = plt.plot([1,1],'green')
plt.legend((hB, hO, hG), ('ACCG', 'TCCG', 'greedy'), frameon=True, loc=(1.04,0))
hB.set_visible(False)
hO.set_visible(False)
hG.set_visible(False)

ax.set_xticklabels(list(range(1, len(fail_int))))
ax.set_xticks(xticks)

plt.ylabel('Surveillance Quality')
plt.xlabel('Event')
plt.savefig(dir + '/surveillance_quality_bp.png', bbox_inches='tight')
plt.clf()


# Plot Errors
base_errors = None
error_diffs = {}
# for opt in ['base', 'agent', 'team', 'greedy', 'random']:
for opt in ['base', 'agent', 'team', 'greedy']:
    time = []
    all_errors = []
    for i in range(5):
        if opt == 'team':
            errors = pd.read_csv(dir_team + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
        else:
            errors = pd.read_csv(dir + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
        errors['max_error'] = errors[[str(n) for n in range(num_nodes)]].max(axis=1)
        time.extend(errors['time'].values)
        all_errors.extend(errors['max_error'].values)
    df = pd.DataFrame([time, all_errors])
    df = df.transpose()
    df.columns = ['time', 'errors']
    df_group = df.groupby('time').agg({'errors': 'mean'}).reset_index()

    df_group = df_group[df_group['time'] > 30]

    if opt == 'base':
        base_errors = df_group
        continue
    else:
        df_group['diff'] = base_errors['errors'].values - df_group['errors'].values
        error_diffs[opt] = df_group

    plt.plot(df_group['time'], df_group['errors'], label=opt)
plt.legend()
plt.savefig(dir + '/errors.png', bbox_inches='tight')
plt.clf()


plt.figure(figsize=[8,4])
ax = plt.axes()
num_fail = 1
xticks = []
for f in range(1, len(fail_int)):
    all_errors = {}
    for i in range(5):
        for opt in ['agent', 'team', 'greedy']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/errors.csv'.format(i=i, opt=opt))

            errors['value'] = errors[[str(n) for n in range(num_nodes)]].mean(axis=1)
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]
            if opt not in all_errors.keys():
                all_errors[opt] = list(errors['value'].values)
            else:
                all_errors[opt].extend(errors['value'].values)

    data = [all_errors['agent'], all_errors['team'], all_errors['greedy']]

    xticks.append(num_fail + 1)
    bp = plt.boxplot(data, positions=[num_fail, num_fail+1, num_fail+2], widths=0.6)
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][1], color='orange')
    plt.setp(bp['boxes'][2], color='green')

    plt.setp(bp['medians'][0], color='blue')
    plt.setp(bp['medians'][1], color='orange')
    plt.setp(bp['medians'][2], color='green')
    num_fail += 4

hB, = plt.plot([1,1],'blue')
hO, = plt.plot([1,1],'orange')
hG, = plt.plot([1,1],'green')
plt.legend((hB, hO, hG), ('ACCG', 'TCCG', 'greedy'), frameon=True, loc=(1.04,0))

hB.set_visible(False)
hO.set_visible(False)
hG.set_visible(False)

ax.set_yscale('log')
ax.set_xticklabels(list(range(1, len(fail_int))))
ax.set_xticks(xticks)
plt.ylabel('Estimation Errors')
plt.xlabel('Event')

plt.savefig(dir + '/errors_bp.png', bbox_inches='tight')
plt.clf()


# Plot Covariance
# for opt in ['agent', 'team', 'greedy', 'random']:
for opt in ['agent', 'team', 'greedy']:
    time = []
    all_errors = []
    for i in range(5):
        if opt == 'team':
            errors = pd.read_csv(dir_team + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))
        else:
            errors = pd.read_csv(dir + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))
        time.extend(errors['time'].values)
        all_errors.extend(errors['value'].values)
    df = pd.DataFrame([time, all_errors])
    df = df.transpose()
    df.columns = ['time', 'errors']
    df_group = df.groupby('time').agg({'errors': 'mean'}).reset_index()

    df_group = df_group[df_group['time'] > 30]

    plt.plot(df_group['time'], df_group['errors'], label=opt)
plt.legend()
plt.savefig(dir + '/max_tr_cov.png', bbox_inches='tight')
plt.clf()


ax = plt.axes()
num_fail = 1
xticks = []
for f in range(1, len(fail_int)):
    all_errors = {}
    for i in range(5):
        for opt in ['agent', 'team', 'greedy']:
            if opt == 'team':
                errors = pd.read_csv(dir_team + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))
            else:
                errors = pd.read_csv(dir + '/{i}_{opt}/max_tr_cov.csv'.format(i=i, opt=opt))

            errors['value'] = errors['value'].abs()
            # errors['value'] = np.log(errors['value'])
            errors = errors[(errors['time'] >= fail_int[f] - fail_freq) & (errors['time'] < fail_int[f])]
            if opt not in all_errors.keys():
                all_errors[opt] = list(errors['value'].values)
            else:
                all_errors[opt].extend(errors['value'].values)

    data = [all_errors['agent'], all_errors['team'], all_errors['greedy']]

    xticks.append(num_fail + 1)
    bp = plt.boxplot(data, positions=[num_fail, num_fail+1, num_fail+2], widths=0.6)
    plt.setp(bp['boxes'][0], color='blue')
    plt.setp(bp['boxes'][1], color='orange')
    plt.setp(bp['boxes'][2], color='green')

    plt.setp(bp['medians'][0], color='blue')
    plt.setp(bp['medians'][1], color='orange')
    plt.setp(bp['medians'][2], color='green')
    num_fail += 4

hB, = plt.plot([1,1],'blue')
hO, = plt.plot([1,1],'orange')
hG, = plt.plot([1,1],'green')
plt.legend((hB, hO, hG), ('ACCG', 'TCCG', 'greedy'), frameon=True, loc=(1.04,0))
hB.set_visible(False)
hO.set_visible(False)
hG.set_visible(False)

ax.set_yscale('log')

ax.set_xticklabels(list(range(1, len(fail_int))))
ax.set_xticks(xticks)

plt.ylabel('Max Trace of Covariance')
plt.xlabel('Event')

plt.savefig(dir + '/max_tr_cov_bp.png', bbox_inches='tight')
plt.clf()



# Calculate Edge Densities
density_dfs = {}
num_possible_edges = (num_nodes * (num_nodes - 1)) / 2
# for opt in ['agent', 'team', 'greedy', 'random']:
for opt in ['agent', 'team', 'greedy']:
    time = []
    edge_density = []
    for i in range(5):
        edge_count = 6
        for t in range(total_time_steps):
            if t in fail_int:
                if opt == 'team':
                    fname = dir_team + '/{i}_{opt}/topologies/{t}.csv'.format(i=i, opt=opt, t=t)
                else:
                    fname = dir + '/{i}_{opt}/topologies/{t}.csv'.format(i=i, opt=opt, t=t)

                new_A = []
                with open(fname, 'r') as f:
                    readCSV = csv.reader(f, delimiter=',')
                    for row in readCSV:
                        data = list(map(float, row))
                        new_A.append(data)

                new_edge_count = 0
                new_A = np.array(new_A)
                for n in range(num_nodes):
                    for l in range(n+1, num_nodes):
                        if new_A[n, l] == 1:
                            new_edge_count += 1
                edge_count = new_edge_count

            time.append(t)
            edge_density.append(edge_count / float(num_possible_edges))

    # Create dataframe for average edge density
    df = pd.DataFrame([time, edge_density])
    df = df.transpose()
    df.columns = ['time', 'edge_density']
    df_group = df.groupby('time').agg({'edge_density': 'mean'}).reset_index()

    df_group = df_group[df_group['time'] > 30]

    density_dfs[opt] = df_group

# Plot Edge Densities against diffs
# for opt in ['agent', 'team', 'greedy', 'random']:
for opt in ['agent', 'team', 'greedy']:
    e = density_dfs[opt]['edge_density'].values
    d = error_diffs[opt]['diff'].values
    plt.scatter(e, d, label=opt)
plt.legend()
plt.savefig(dir + '/edge_density.png')
plt.clf()


# Plot Targets
# for opt in ['agent', 'team', 'greedy', 'random']:
for opt in ['agent', 'team', 'greedy']:
    if opt == 'team':
        est = pd.read_csv(dir_team + '/0_{opt}/estimates.csv'.format(opt=opt))
        robot_pos = pd.read_csv(dir_team + '/0_{opt}/robot_positions.csv'.format(opt=opt))
    else:
        est = pd.read_csv(dir + '/0_{opt}/estimates.csv'.format(opt=opt))
        robot_pos = pd.read_csv(dir + '/0_{opt}/robot_positions.csv'.format(opt=opt))
    colors = ['red', 'blue', 'green', 'orange']
    tops = [0] + fail_int
    for i in tops:
        ax = plt.axes()
        rs = robot_pos[robot_pos['time'] == i]
        for t in range(4):
            df = pd.read_csv(dir + '/0_{opt}/target_{t}_positions.csv'.format(opt=opt, t=t))
            tmp = df.loc[i - 5 + 1: i + 2]

            plt.plot(tmp['x'].values, tmp['y'].values, '+',
                     label="True Target {t}".format(t=t), alpha=0.8, color=colors[t])

            e = est[est['target'] == t]
            e = e.groupby('time').agg({'x': 'mean', 'y': 'mean'}).reset_index()
            e = e[(e['time'] >= i - 5) & (e['time'] <= i+1)]
            plt.plot(e['x'].values, e['y'].values, '--',
                     label="Estimate {t}".format(t=t), alpha=0.5, color=colors[t])

        plt.scatter(rs['x'].values, rs['y'].values, color='black', marker='x')

        # Plot Adjacencies
        edge_list = []
        new_A = []
        if opt == 'team':
            topology_file = dir_team + '/0_{opt}/topologies/{i}.csv'.format(opt=opt, i=i)
        else:
            topology_file = dir + '/0_{opt}/topologies/{i}.csv'.format(opt=opt, i=i)
        with open(topology_file, 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
            for row in readCSV:
                data = list(map(float, row))
                new_A.append(data)
        new_A = np.array(new_A)
        num_drones = new_A.shape[0]
        for n in range(num_drones):
            for o in range(n+1, num_drones):
                if new_A[n, o] == 1:
                    n_pos = rs[rs['node_id'] == n]
                    o_pos = rs[rs['node_id'] == o]

                    xl = [n_pos['x'].values[0], o_pos['x'].values[0]]
                    yl = [n_pos['y'].values[0], o_pos['y'].values[0]]
                    plt.plot(xl, yl, color='gray', alpha=0.5)

            # Plot FOV
            tmp_rs = rs[rs['node_id'] == n]
            p = plt.Circle((tmp_rs['x'].values[0], tmp_rs['y'].values[0]), tmp_rs['fov_radius'].values[0], alpha=0.1)
            ax.add_patch(p)

        plt.xlim([-50, 50])
        plt.ylim([-50, 50])
        # plt.legend()
        plt.savefig(dir + '/0_{opt}/overhead/{i}.png'.format(opt=opt, i=i))
        plt.clf()
