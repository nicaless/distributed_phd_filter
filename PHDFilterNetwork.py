from copy import deepcopy
import math
import networkx as nx
import numpy as np
from operator import attrgetter

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
        self.cardinality = 0
        self.node_keep = {}

        # TRACKERS
        self.adjacencies = {}
        self.weighted_adjacencies = {}

    def step_through(self, measurements, L=1, how='geom'):
        nodes = nx.get_node_attributes(self.network, 'node')
        if not isinstance(measurements, dict):
            measurements = {0: measurements}
        for i, m in measurements.items():
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
                n.update_trackers(i, pre_consensus=False)

            self.adjacencies[i] = self.adjacency_matrix()
            self.weighted_adjacencies[i] = self.weighted_adjacency_matrix()

    def cardinality_consensus(self):
        nodes = nx.get_node_attributes(self.network, 'node')
        weights = nx.get_node_attributes(self.network, 'weights')

        weighted_est = 0
        tot_est = []
        for n in list(self.network.nodes()):
            est = sum([t.weight for t in nodes[n].targets])
            weighted_est += est * weights[n]
            tot_est.append(est)

        # Rescale Weights using total weighted estimate
        for n in list(self.network.nodes()):
            for t in nodes[n].targets:
                t.weight = np.ceil(weighted_est) / tot_est[n]

        self.cardinality = int(np.ceil(weighted_est))

    def reduce_comps(self, node_id):
        node = nx.get_node_attributes(self.network, 'node')[node_id]
        node_comps = node.targets
        limit = min(self.cardinality, node.max_components)
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
                y_weight = nx.get_node_attributes(self.network, 'weights')[neighbor]
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
            if how == 'geom':
                new_comps = self.fuse_comps_geom(n)
            else:
                new_comps = self.fuse_comps_arith(n)
            new_comps.sort(key=attrgetter('weight'), reverse=True)
            node = nx.get_node_attributes(self.network, 'node')[n]
            # node.targets = new_comps

            node.updated_targets = new_comps
            node.prune()
            node.merge()
            node.targets = node.merged_targets

    def fuse_comps_arith(self, node_id):
        node_comps = self.node_share[node_id]['node_comps']

        new_covs = self.fuse_covs(node_id, how='arith')
        new_states, did_fuse = self.fuse_states(node_id, how='arith')
        new_alphas = self.fuse_alphas(node_id, new_covs, new_states, how='arith')

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
                sum_covs = weight * comp_cov_inv
            else:
                sum_covs = weight * node_comps[i].state_cov

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
                sum_states = weight * np.dot(comp_cov_inv, comp_state)
            else:
                sum_states = weight * comp_state
                sum_weights = weight

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
                all_comps = [(weight, node_comps[i])] + closest_neighbor_comps[i]
                K = self.calcK(i, all_comps, new_covs, new_states)
                comp_cov = node_comps[i].state_cov
                rescaler = self.rescaler(comp_cov, weight)
                prod_alpha = comp_alpha ** weight * rescaler * K
            else:
                sum_alpha = comp_alpha

            for c in closest_neighbor_comps[i]:
                n_weight = c[0]
                n_comp = c[1]
                n_comp_alpha = n_comp.weight
                if how == 'geom':
                    n_comp_cov = n_comp.state_cov
                    rescaler = self.rescaler(n_comp_cov, n_weight)
                    prod_alpha *= n_comp_alpha ** n_weight * rescaler * K
                else:
                    sum_alpha += n_comp_alpha
            if how == 'geom':
                new_alphas.append(prod_alpha)
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
            A[n, n] = weights[n]
            for i, neighbor in enumerate(list(G.neighbors(n))):
                A[n, neighbor] = weights[neighbor]

        return A


















