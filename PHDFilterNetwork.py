import math
import networkx as nx
import numpy as np
from operator import attrgetter
import pandas as pd
import scipy

from optimization_utils import *
from ospa import *
from reconfig_utils import *
from target import Target


class PHDFilterNetwork:
    def __init__(self,
                 nodes,
                 weights,
                 G,
                 region=[(-50, 50), (-50, 50)]
                 ):
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, weights, 'weights')
        self.region = region
        self.target_estimates = []

        self.node_share = {}
        self.cardinality = {}
        self.node_keep = {}

        # TRACKERS
        self.failures = {}
        self.adjacencies = {}
        self.weighted_adjacencies = {}
        self.errors = {}
        self.max_trace_cov = {}
        self.mean_trace_cov = {}
        self.gospa = {}
        self.nmse_card = {}

    def step_through(self, measurements, true_targets,
                     L=3, how='geom', opt='agent',
                     fail_int=None, fail_sequence=None,
                     base=False, noise_mult=1):
        nodes = nx.get_node_attributes(self.network, 'node')
        if not isinstance(measurements, dict):
            measurements = {0: measurements}
        failure = False
        new_metro_weights = False
        for i, m in measurements.items():
            if fail_int is not None or fail_sequence is not None:
                if fail_int is not None:
                    if i in fail_int:
                        failure = True
                        fail_node = self.apply_failure(i, mult=noise_mult)
                else:
                    if i in fail_sequence:
                        failure = True
                        fail_node = self.apply_failure(i, fail=fail_sequence[i])

            for id, n in nodes.items():
                n.step_through(m, i)

            self.cardinality_consensus()

            for id, n in nodes.items():
                self.reduce_comps(id)

            for l in range(L):
                for id, n in nodes.items():
                    self.share_info(id)
                    self.get_closest_comps(id)
                self.update_comps(how=how)
                for id, n in nodes.items():
                    self.reduce_comps(id)

            min_card = min([c for i, c in self.cardinality.items()])
            min_targets = min([len(n.targets) for i, n in nodes.items()])
            min_targets = min(min_targets, min_card)
            covariance_matrix = self.get_covariance_matrix(min_targets,
                                                           true_targets[i])
            c_data, c_tr, c_inv_tr = self.get_covariance_trace(covariance_matrix,
                                                               how=how)
            if failure and not base:
                A = self.adjacency_matrix()
                current_coords = {nid: n.position for nid, n in nodes.items()}
                fov = {nid: n.fov for nid, n in nodes.items()}

                weights = nx.get_node_attributes(self.network, 'weights')

                if opt == 'agent':
                    A, new_weights = agent_opt(A, weights, c_data,
                                               failed_node=fail_node)
                elif opt == 'greedy':
                    # trace of cov of non-neighbors
                    c_tr_copy = deepcopy(c_tr)
                    non_neighbors = list(nx.non_neighbors(self.network, fail_node))
                    c_tr_copy[fail_node] = np.inf

                    c_tr_copy_fill = []
                    for n, val in enumerate(c_tr_copy):
                        if n in non_neighbors:
                            c_tr_copy_fill.append(val)
                        else:
                            c_tr_copy_fill.append(np.inf)

                    best_node = c_tr_copy.index(min(c_tr_copy))
                    A[best_node, fail_node] = 1
                    A[fail_node, best_node] = 1
                    new_metro_weights = True
                elif opt == 'random':
                    non_neighbors = list(nx.non_neighbors(self.network, fail_node))
                    if len(non_neighbors) != 0:
                        rand_node = np.random.choice(non_neighbors)
                        A[rand_node, fail_node] = 1
                        A[fail_node, rand_node] = 1
                        new_metro_weights = True
                else:
                    if how == 'geom':
                        inv_cov_mat = []
                        for c in covariance_matrix:
                            inv_cov_mat.append(np.linalg.inv(c))
                        A, new_weights = team_opt(A, weights, inv_cov_mat)
                    else:
                        A, new_weights = team_opt(A, weights, covariance_matrix)

                    # A, new_weights = team_opt2(A, weights, covariance_matrix)

                G = nx.from_numpy_matrix(A)
                self.network = G
                nx.set_node_attributes(self.network, nodes, 'node')
                if new_metro_weights:
                    new_weights = {}
                    for id in range(len(list(nodes.keys()))):
                        new_weights[id] = {}
                        self_degree = G.degree(id)
                        metropolis_weights = []
                        for n in G.neighbors(id):
                            degree = G.degree(n)
                            mw = 1 / (1 + max(self_degree, degree))
                            new_weights[id][n] = mw
                            metropolis_weights.append(mw)
                        new_weights[id][id] = 1 - sum(metropolis_weights)
                    new_metro_weights = False
                nx.set_node_attributes(self.network, new_weights, 'weights')

                centroid = self.get_centroid()
                new_coords = generate_coords(A, current_coords, fov, centroid)
                if new_coords:
                    for id, n in nodes.items():
                        n.update_position(new_coords[id])
                failure = False

            for id, n in nodes.items():
                n.update_trackers(i, pre_consensus=False)

            self.max_trace_cov[i] = max(c_tr)
            self.mean_trace_cov[i] = np.mean(c_tr)
            self.errors[i] = self.calc_errors(true_targets[i])
            self.gospa[i] = self.calc_ospa(true_targets[i])
            self.nmse_card[i] = self.calc_nmse_card(true_targets[i])
            self.adjacencies[i] = self.adjacency_matrix()
            self.weighted_adjacencies[i] = self.weighted_adjacency_matrix()

    def apply_failure(self, i, fail=None, mult=1):
        nodes = nx.get_node_attributes(self.network, 'node')

        # Generate new R
        if fail is None:
            fail_node = np.random.choice(list(nodes.keys()))

            # Get R from Node
            R = nodes[fail_node].R

            r_mat_size = R.shape[0]
            r = scipy.random.rand(r_mat_size, r_mat_size) * mult
            rpd = np.dot(r, r.T)
        else:
            fail_node = fail[0]
            rpd = fail[1]
            R = nodes[fail_node].R

        R = R + rpd
        nodes[fail_node].R = R
        self.failures[i] = (fail_node, rpd)
        return fail_node

    def get_covariance_trace(self, cov_matrix, how='geom'):
        nodes = nx.get_node_attributes(self.network, 'node')

        covariance_matrix = cov_matrix

        covariance_data = []
        cov_tr = []
        inv_cov_tr = []
        for n, node in nodes.items():
            cov = covariance_matrix[n]
            tr_cov = np.trace(cov)
            tr_inv_cov = np.trace(np.linalg.inv(cov +
                                                np.eye(cov.shape[0]) * 10e-6))
            cov_tr.append(tr_cov)
            inv_cov_tr.append(tr_inv_cov)
            if how == 'geom':
                if tr_inv_cov == 0:
                    tr_inv_cov = 0.01 * np.random.random(1)[0]
                d = tr_inv_cov

            else:
                d = tr_cov
            if self.cardinality[n] == 0:
                d = 0
            else:
                d = d / float(self.cardinality[n])
            covariance_data.append(d)

        return covariance_data, cov_tr, inv_cov_tr

    def get_covariance_matrix(self, min_targets, true_targets):
        nodes = nx.get_node_attributes(self.network, 'node')

        keep_targets = true_targets[:min_targets]

        all_covs = []
        for n, node in nodes.items():
            cov = np.zeros((4 * len(keep_targets), 4 * len(keep_targets)))
            all_covs.append(cov)

        for i, t in enumerate(keep_targets):
            for n, node in nodes.items():
                distances = [math.hypot(t[0] - comp.state[0][0],
                                        t[1] - comp.state[1][0])
                             for comp in node.targets]
                min_index = distances.index(min(distances))
                all_covs[n][i:i+4, i:i+4] = node.targets[min_index].state_cov
        return all_covs

    def get_centroid(self):
        nodes = nx.get_node_attributes(self.network, 'node')
        all_phd_states = []
        for n, node in nodes.items():
            for t in node.targets:
                pos = np.array([[t.state[0][0]], [t.state[1][0]]])
                all_phd_states.append(pos)

        if len(all_phd_states) == 0:
            return np.array([[0], [0]])

        x, y = zip(*all_phd_states)
        center_x = sum(x) / float(len(x))
        center_y = sum(y) / float(len(x))
        return np.array([[center_x], [center_y]])

    def cardinality_consensus(self):
        nodes = nx.get_node_attributes(self.network, 'node')
        weights = nx.get_node_attributes(self.network, 'weights')

        for n1 in list(self.network.nodes()):
            weighted_est = 0
            tot_est = {}
            for n2, weight in weights[n1].items():
                est = sum([t.weight for t in nodes[n2].targets])
                weighted_est += est * weight
                tot_est[n2] = est

            # Rescale Weights using total weighted estimate
            for t in nodes[n1].targets:
                t.weight *= np.ceil(weighted_est) / tot_est[n1]

            self.cardinality[n1] = int(np.ceil(weighted_est))

    def reduce_comps(self, node_id):
        node = nx.get_node_attributes(self.network, 'node')[node_id]
        node_comps = node.targets
        limit = min(self.cardinality[node_id], node.max_components)
        self.cardinality[node_id] = limit
        keep_node_comps = node_comps[:limit]
        node.targets = keep_node_comps

    def share_info(self, node_id):
        G = self.network

        node = nx.get_node_attributes(G, 'node')[node_id]
        self.node_share[node_id] = {}
        self.node_share[node_id]['node_comps'] = node.targets
        neighbor_comps = {}
        for neighbor_id in list(G.neighbors(node_id)):
            neighbor = nx.get_node_attributes(G, 'node')[neighbor_id]
            neighbor_comps[neighbor_id] = neighbor.targets

        self.node_share[node_id]['neighbor_comps'] = neighbor_comps

    def get_closest_comps(self, node_id):
        node_comps = self.node_share[node_id]['node_comps']
        neighbor_comps = self.node_share[node_id]['neighbor_comps']
        merge_thresh = nx.get_node_attributes(self.network, 'node')[node_id].merge_thresh

        keep_comps = {}
        for i in range(len(node_comps)):
            keep_comps[i] = []
            x_state = node_comps[i].state
            x_cov = node_comps[i].state_cov
            for neighbor, n_comps in neighbor_comps.items():
                y_weight = nx.get_node_attributes(self.network,
                                                  'weights')[node_id][neighbor]
                d = merge_thresh
                for neighbor_comp in n_comps:
                    y_state = neighbor_comp.state
                    # new_d = float(np.dot(np.dot((x_state - y_state).T,
                    #                         np.linalg.inv(x_cov)),
                    #                  x_state - y_state))
                    new_d = math.hypot(y_state[0][0] - x_state[0][0],
                                       y_state[1][0] - x_state[1][0])
                    if new_d < d:
                        current_closest = neighbor_comp
                        keep_comps[i].append((y_weight, current_closest))

        self.node_keep[node_id] = keep_comps

    def update_comps(self, how='geom'):
        for n in list(self.network.nodes()):
            node = nx.get_node_attributes(self.network, 'node')[n]
            if how == 'geom':
                new_comps = self.fuse_comps_geom(n)
            else:
                new_comps = self.fuse_comps_arith(n)
            new_comps.sort(key=attrgetter('weight'), reverse=True)

            if np.isnan([comp.state for comp in new_comps]).any():
                print('nan state after fusing comps')

            tot_est = sum([comp.weight for comp in new_comps])
            # Rescale Weights using cardinality
            for t in new_comps:
                t.weight *= self.cardinality[n] / tot_est

            node.updated_targets = new_comps
            node.prune()
            node.merge()
            node.targets = node.merged_targets

    def fuse_comps_arith(self, node_id):
        node_comps = self.node_share[node_id]['node_comps']

        new_covs = self.fuse_covs(node_id, how='arith')
        new_states, did_fuse = self.fuse_states(node_id, how='arith')
        if np.isnan(new_states).any():
            print('nan states after fusion')
        new_alphas = self.fuse_alphas(node_id, new_covs, new_states, how='arith')
        if np.isnan(new_alphas).any():
            print('nan alphas after fusion')

        fused_comps = []
        for i in range(len(new_alphas)):
            f = Target(init_weight=new_alphas[i],
                       init_state=new_states[i],
                       init_cov=new_covs[i],
                       dt_1=node_comps[i].dt_1,
                       dt_2=node_comps[i].dt_2)
            fused_comps.append(f)
        return fused_comps

    # After fusing covs, states, and alphas the geometric way
    def fuse_comps_geom(self, node_id):
        node_comps = self.node_share[node_id]['node_comps']

        new_covs = self.fuse_covs(node_id)
        new_states, did_fuse = self.fuse_states(node_id)
        new_alphas = self.fuse_alphas(node_id, new_covs, new_states)
        for i in range(len(new_states)):
            if did_fuse[i] == 1:
                n = np.dot(new_covs[i], new_states[i])
                new_states[i] = n

        fused_comps = []
        for i in range(len(new_alphas)):
            f = Target(init_weight=new_alphas[i],
                       init_state=new_states[i],
                       init_cov=new_covs[i],
                       dt_1=node_comps[i].dt_1,
                       dt_2=node_comps[i].dt_2)
            fused_comps.append(f)
        return fused_comps

    def fuse_covs(self, node_id, how='geom'):
        # Use self.node_keep to fuse only the closest components among all neighbors
        # If no close components skip fusion for this component

        node_comps = self.node_share[node_id]['node_comps']
        closest_neighbor_comps = self.node_keep[node_id]

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        new_covs = []
        for i in range(len(node_comps)):
            if len(closest_neighbor_comps[i]) == 0:
                new_covs.append(node_comps[i].state_cov)
                continue

            if how == 'geom':
                comp_cov_inv = np.linalg.inv(node_comps[i].state_cov)
                sum_covs = weight[node_id] * comp_cov_inv
            else:
                sum_covs = weight[node_id] * node_comps[i].state_cov

            for c in closest_neighbor_comps[i]:
                n_weight = c[0]
                n_comp = c[1]
                if how == 'geom':
                    n_comp_cov_inv = np.linalg.inv(n_comp.state_cov)
                    sum_covs += n_weight * n_comp_cov_inv
                else:
                    sum_covs += n_weight * n_comp.state_cov
            if how == 'geom':
                sum_covs = sum_covs + np.eye(sum_covs.shape[0]) * 10e-6
                new_covs.append(np.linalg.inv(sum_covs))
            else:
                new_covs.append(sum_covs)

        return new_covs

    def fuse_states(self, node_id, how='geom'):
        # Use self.node_keep to fuse only the closest components among all neighbors
        # If no close components skip fusion for this component

        node_comps = self.node_share[node_id]['node_comps']
        closest_neighbor_comps = self.node_keep[node_id]

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        new_states = []
        did_fuse = []
        for i in range(len(node_comps)):
            if len(closest_neighbor_comps[i]) == 0:
                new_states.append(node_comps[i].state)
                did_fuse.append(0)
                continue

            did_fuse.append(1)
            comp_state = node_comps[i].state
            if how == 'geom':
                comp_cov_inv = np.linalg.inv(node_comps[i].state_cov)
                sum_states = weight[node_id] * np.dot(comp_cov_inv, comp_state)
            else:
                sum_states = weight[node_id] * comp_state
            sum_weights = weight[node_id]

            for c in closest_neighbor_comps[i]:
                n_weight = c[0]
                n_comp = c[1]
                n_comp_state = n_comp.state
                if how == 'geom':
                    n_comp_cov_inv = np.linalg.inv(n_comp.state_cov)
                    sum_states += n_weight * np.dot(n_comp_cov_inv, n_comp_state)
                else:
                    sum_states += n_weight * n_comp_state
                sum_weights += n_weight

            if sum_weights == 0:
                new_states.append(node_comps[i].state)
                did_fuse.append(0)
            else:
                if how == 'geom':
                    new_states.append(sum_states)
                else:
                    new_states.append(sum_states / float(sum_weights))

        return new_states, did_fuse

    # Fuse the component weights
    def fuse_alphas(self, node_id, new_covs, new_states, how='geom'):
        # Use self.node_keep to fuse only the closest components among all neighbors
        # If no close components skip fusion for this component

        node_comps = self.node_share[node_id]['node_comps']
        closest_neighbor_comps = self.node_keep[node_id]

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        new_alphas = []
        for i in range(len(node_comps)):
            if len(closest_neighbor_comps[i]) == 0:
                new_alphas.append(node_comps[i].weight)
                continue

            comp_alpha = node_comps[i].weight
            if how == 'geom':
                all_comps = [(weight[node_id], node_comps[i])] + \
                            closest_neighbor_comps[i]
                K = self.calcK(i, all_comps, new_covs, new_states)
                comp_cov = node_comps[i].state_cov
                rescaler = self.rescaler(comp_cov, weight[node_id])
                # prod_alpha = comp_alpha ** weight * rescaler * K
                prod_alpha = (comp_alpha ** weight[node_id]) * rescaler
            else:
                sum_alpha = comp_alpha

            for c in closest_neighbor_comps[i]:
                n_weight = c[0]
                n_comp = c[1]
                n_comp_alpha = n_comp.weight
                if how == 'geom':
                    n_comp_cov = n_comp.state_cov
                    rescaler = self.rescaler(n_comp_cov, n_weight)
                    prod_alpha *= (n_comp_alpha ** n_weight) * rescaler
                else:
                    sum_alpha += n_comp_alpha

            if how == 'geom':
                new_alphas.append(prod_alpha * K)
            else:
                new_alphas.append(sum_alpha)

        return new_alphas

    @staticmethod
    def calcK(comp_id, all_comps, new_covs, new_states):
        # Kappa 1:n
        firstterm_1n = len(all_comps) * 4 * np.log(2 * np.pi)
        secondterm_1n = 0
        thirdterm_1n = 0

        # Kappa n
        firstterm_n = 4 * np.log(2 * np.pi)
        secondterm_n = 0
        thirdterm_n = np.dot(np.dot(new_states[comp_id].T, new_covs[comp_id]),
                             new_states[comp_id])

        for i in range(len(all_comps)):
            c = all_comps[i]
            weight = c[0]
            state = c[1].state
            cov = c[1].state_cov
            # cov = np.eye(cov.shape[0]) * 10e-6
            secondterm_1n += np.log(np.linalg.det(weight * np.linalg.inv(cov)))
            thirdterm_1n += weight * np.dot(np.dot(state.T,
                                                   np.linalg.inv(cov)),
                                            state)
            secondterm_n += np.linalg.det(weight * np.linalg.inv(cov))

        kappa_1n = -0.5 * (firstterm_1n - secondterm_1n + thirdterm_1n)
        kappa_n = -0.5 * (firstterm_n - np.log(secondterm_n) + thirdterm_n)

        K = np.exp(kappa_1n[0][0] - kappa_n[0][0])
        return K

    @staticmethod
    def rescaler(cov, weight):
        numer = np.linalg.det(2 * np.pi * np.linalg.inv(cov / weight))
        denom = np.linalg.det(2 * np.pi * cov) ** weight
        return (numer / denom) ** 0.5

    def adjacency_matrix(self):
        G = self.network
        num_nodes = len(list(G.nodes()))

        A = nx.adjacency_matrix(G).todense()
        if not np.array_equal(np.diag(A), np.ones(num_nodes)):
            A = A + np.diag(np.ones(num_nodes))
        return A

    def weighted_adjacency_matrix(self):
        A = self.adjacency_matrix()
        G = self.network

        for n in list(G.nodes()):
            weights = nx.get_node_attributes(G, 'weights')
            A[n, n] = weights[n][n]
            for i, neighbor in enumerate(list(G.neighbors(n))):
                A[n, neighbor] = weights[n][neighbor]

        return A

    def calc_errors(self, true_targets):
        nodes = nx.get_node_attributes(self.network, 'node')

        max_errors = []
        for i, t in enumerate(true_targets):
            errors = []
            for n, node in nodes.items():
                if not node.check_measure_oob(t) and len(node.targets) > 0:
                    distances = [math.hypot(t[0] - comp.state[0][0],
                                            t[1] - comp.state[1][0])
                                 for comp in node.targets]
                    errors.append(min(distances))
            if len(errors) == 0:
                max_errors.append(0)
            else:
                max_errors.append(max(errors))
        return np.max(max_errors)

    def calc_ospa(self, true_targets):
        tracks = []
        nodes = nx.get_node_attributes(self.network, 'node')

        for n, node in nodes.items():
            for t in node.targets:
                tracks.append(np.array([[t.state[0][0]],
                                        [t.state[1][0]]]))

        gospa, \
        target_to_track_assigments, \
        gospa_localization, \
        gospa_missed, \
        gospa_false = calculate_gospa(true_targets, tracks)

        return gospa

    def calc_nmse_card(self, true_targets):
        N = len(true_targets)
        squared_error = 0
        sum_card = 0

        for n, card in self.cardinality.items():
            squared_error += (card - N) ** 2
            sum_card += card

        return squared_error / (N * sum_card)

    def save_metrics(self, path):
        # Save Errors
        errors = pd.DataFrame.from_dict(self.errors, orient='index')
        errors.columns = ['value']
        errors.to_csv(path + '/errors.csv', index_label='time')

        # Save Max Trace Cov
        max_tr_cov = pd.DataFrame.from_dict(self.max_trace_cov, orient='index')
        max_tr_cov.columns = ['value']
        max_tr_cov.to_csv(path + '/max_tr_cov.csv', index_label='time')

        # Save Mean Trace Cov
        mean_tr_cov = pd.DataFrame.from_dict(self.mean_trace_cov, orient='index')
        mean_tr_cov.columns = ['value']
        mean_tr_cov.to_csv(path + '/mean_tr_cov.csv', index_label='time')

        # Save OSPA
        ospa = pd.DataFrame.from_dict(self.gospa, orient='index')
        ospa.columns = ['value']
        ospa.to_csv(path + '/ospa.csv', index_label='time')

        # Save NMSE
        nmse = pd.DataFrame.from_dict(self.nmse_card, orient='index')
        nmse.columns = ['value']
        nmse.to_csv(path + '/nmse.csv', index_label='time')

    def save_estimates(self, path):
        all_nodes = nx.get_node_attributes(self.network, 'node')
        time = []
        x = []
        y = []
        for n, node in all_nodes.items():
            for t, pos in node.consensus_positions.items():
                for p in pos:
                    time.append(t)
                    x.append(p[0][0])
                    y.append(p[1][0])
        data = pd.DataFrame([time, x, y])
        data = data.transpose()
        data.columns = ['time', 'x', 'y']
        data.to_csv(path + '/estimates.csv', index=False)

    def save_positions(self, path):
        time = []
        x = []
        y = []
        z = []
        fov = []
        node_id = []
        all_nodes = nx.get_node_attributes(self.network, 'node')
        for n, node in all_nodes.items():
            for t, pos in node.node_positions.items():
                time.append(t)
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                fov.append(node.fov)
                node_id.append(n)
        data = pd.DataFrame([time, x, y, z, fov, node_id])
        data = data.transpose()
        data.columns = ['time', 'x', 'y', 'z', 'fov_radius', 'node_id']
        data.to_csv(path + '/robot_positions.csv', index=False)

    def save_topologies(self, path):
        for t, a in self.adjacencies.items():
            np.savetxt(path + '/{t}.csv'.format(t=t),
                       a, delimiter=',')
























