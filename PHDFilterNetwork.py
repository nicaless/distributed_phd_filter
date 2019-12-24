import math
import networkx as nx
import numpy as np
from operator import attrgetter

from optimization_utils import *
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
        self.adjacencies = {}
        self.weighted_adjacencies = {}

    def step_through(self, measurements, true_targets, L=1, how='geom', opt='agent'):
        nodes = nx.get_node_attributes(self.network, 'node')
        if not isinstance(measurements, dict):
            measurements = {0: measurements}
        failure = False
        for i, m in measurements.items():
            # TODO: Apply Failure Event
            if i == 10:
                failure = True

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

            if failure:
                A = self.adjacency_matrix()
                current_coords = {nid: n.position for nid, n in nodes.items()}
                fov = {nid: n.fov for nid, n in nodes.items()}

                weights = nx.get_node_attributes(self.network, 'weights')
                if opt == 'agent':
                    covariance_data = self.get_covariance_trace(true_targets[i],
                                                                how=how)
                    A, new_weights = agent_opt(A, weights, covariance_data,
                                               failed_node=0)
                else:
                    # TODO: implement global optimization
                    covariance_data = self.get_covariance_trace(true_targets[i])
                    A, new_weights = agent_opt(A, weights, covariance_data,
                                               failed_node=0)

                G = nx.from_numpy_matrix(A)
                self.network = G
                nx.set_node_attributes(self.network, nodes, 'node')
                nx.set_node_attributes(self.network, new_weights, 'weights')

                centroid = self.get_centroid()
                new_coords = generate_coords(A, current_coords, fov, centroid)
                for id, n in nodes.items():
                    n.update_position(new_coords[id])
                failure = False

            for id, n in nodes.items():
                n.update_trackers(i, pre_consensus=False)

            self.adjacencies[i] = self.adjacency_matrix()
            self.weighted_adjacencies[i] = self.weighted_adjacency_matrix()

    def get_covariance_trace(self, true_targets, how='geom'):
        nodes = nx.get_node_attributes(self.network, 'node')
        min_card = min([c for i, c in self.cardinality.items()])
        min_targets = min([len(n.targets) for i, n in nodes.items()])
        min_targets = min(min_targets, min_card)

        covariance_matrix = self.get_covariance_matrix(min_targets,
                                                       true_targets)
        covariance_data = []
        for n, node in nodes.items():
            if how == 'geom':
                # d = np.trace(np.linalg.inv(node.targets[0].state_cov))
                d = np.trace(np.linalg.inv(covariance_matrix[n] +
                                           np.eye(covariance_matrix[n].shape[0])
                                           * 10e-6))
            else:
                # d = np.trace(node.targets[0].state_cov)
                d = np.trace(covariance_matrix[n])
            d = d / float(self.cardinality[n])
            covariance_data.append(d)

        return covariance_data

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

        x, y = zip(*all_phd_states)
        center_x = sum(x) / float(len(x))
        center_y = sum(y) / float(len(x))
        return np.array([[center_x], [center_y]])

    # def cardinality_consensus(self):
    #     nodes = nx.get_node_attributes(self.network, 'node')
    #     weights = nx.get_node_attributes(self.network, 'weights')
    #
    #     weighted_est = 0
    #     tot_est = []
    #     for n in list(self.network.nodes()):
    #         est = sum([t.weight for t in nodes[n].targets])
    #         weighted_est += est * weights[n]
    #         tot_est.append(est)
    #
    #     # Rescale Weights using total weighted estimate
    #     for n in list(self.network.nodes()):
    #         for t in nodes[n].targets:
    #             t.weight *= np.ceil(weighted_est) / tot_est[n]
    #
    #     self.cardinality = int(np.ceil(weighted_est))

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
        # limit = min(self.cardinality, node.max_components)
        limit = min(self.cardinality[node_id], node.max_components)
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
                # sum_covs = weight * comp_cov_inv
                sum_covs = weight[node_id] * comp_cov_inv
            else:
                # sum_covs = weight * node_comps[i].state_cov
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
                # sum_states = weight * np.dot(comp_cov_inv, comp_state)
                sum_states = weight[node_id] * np.dot(comp_cov_inv, comp_state)
            else:
                # sum_states = weight * comp_state
                # sum_weights = weight
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
                    # prod_alpha *= n_comp_alpha ** n_weight * rescaler * K
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


















