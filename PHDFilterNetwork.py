import networkx as nx
import numpy as np

from PHDFilterNode import dmvnorm


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
        self.node_keep = {}

    def step_through(self, measurements, folder='results'):
        for id, n in nx.get_node_attributes(self.network, 'node').items():
            n.step_through(measurements, folder='{f}/{id}'.format(f=folder,
                                                                  id=id))

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

        keep_comps = {}
        for i in range(len(node_comps)):
            keep_comps[i] = []
            x_state = node_comps[i].state
            x_cov = node_comps[i].state_cov
            for neighbor, n_comps in neighbor_comps.items():
                y_weight = nx.get_node_attributes(self.network, 'weights')[neighbor]
                d = 1e5
                current_closest = None
                for neighbor_comp in n_comps:
                    y_state = neighbor_comp.state
                    new_d = float(np.dot(np.dot((x_state - y_state).T,
                                            np.linalg.inv(x_cov)),
                                     x_state - y_state))
                    if new_d < d:
                        current_closest = neighbor_comp
                        d = new_d
                if current_closest is not None:
                    keep_comps[i].append((y_weight, current_closest))

        self.node_keep[node_id] = keep_comps

    # Each node to replace targets variable with fusion_targets variable (after L fusion iterations)
    def update_comps(self):
        pass

    # Node to update the fusion_targets variable after completing fusion steps
    def fuse_comps(self, node_id):
        pass

    def fuse_covs(self, node_id):
        # Use self.node_keep to fuse only the closest components among all neighbors
        # If no close components skip fusion for this component

        node_comps = self.node_share[node_id]['node_comps']
        closest_neighbor_comps = self.node_keep[node_id]

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        new_covs = []
        for i in range(len(node_comps)):
            comp_cov_inv = np.linalg.inv(node_comps[i].state_cov)
            sum_covs = weight * comp_cov_inv

            if len(closest_neighbor_comps[i]) == 0:
                new_covs.append(sum_covs)
                continue

            for c in closest_neighbor_comps[i]:
                n_weight = c[0]
                n_comp = c[1]
                n_comp_cov_inv = np.linalg.inv(n_comp.state_cov)
                sum_covs += n_weight * n_comp_cov_inv
            new_covs.append(sum_covs)

        return new_covs

    def fuse_states(self, node_id, new_covs):
        node_comps = self.node_share[node_id]['node_comps']
        neighbor_comps = self.node_share[node_id]['neighbor_comps']

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        # Only going to merge up to the min num of components among neighbors
        min_comp = min([len(node_comps)] +
                       [len(cs) for n, cs in neighbor_comps.items()])

        new_states = []
        for i in range(min_comp):
            comp_cov_inv = np.linalg.inv(node_comps[i].state_cov)
            comp_state = node_comps[i].state
            sum_states = weight * np.dot(comp_cov_inv, comp_state)
            for neighbor, n_comps in neighbor_comps.items():
                n_weight = nx.get_node_attributes(self.network, 'weights')[neighbor]
                n_comp_cov_inv = np.linalg.inv(n_comps[i].state_cov)
                n_comp_state = n_comps[i].state
                sum_states += n_weight * np.dot(n_comp_cov_inv, n_comp_state)
            new_states.append(np.dot(new_covs[i], sum_states))

        return new_states

    # Fuse the component weights
    def fuse_alphas(self, node_id, new_states):
        node_comps = self.node_share[node_id]['node_comps']
        neighbor_comps = self.node_share[node_id]['neighbor_comps']

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        # Only going to merge up to the min num of components among neighbors
        min_comp = min([len(node_comps)] +
                       [len(cs) for n, cs in neighbor_comps.items()])

        new_alphas = []
        for i in range(min_comp):
            comp_cov = node_comps[i].state_cov
            comp_alpha = node_comps[i].weight
            prod_alphas = (comp_alpha ** weight) * self.kappa(comp_cov, weight)
            for neighbor, n_comps in neighbor_comps.items():
                n_weight = nx.get_node_attributes(self.network, 'weights')[neighbor]
                n_comp_cov = n_comps[i].state_cov
                n_comp_alpha = n_comps[i].weight
                prod_alphas *= (n_comp_alpha ** n_weight) * self.kappa(n_comp_cov, weight)

            new_alphas.append(prod_alphas)

            # Evaluate Gaussians
            # for j in range(len(new_states)):

        return new_alphas

    # For fusing the alphas, first need to weight sum the covs
    def weight_sum_covs(self, node_id):
        node_comps = self.node_share[node_id]['node_comps']
        neighbor_comps = self.node_share[node_id]['neighbor_comps']

        weight = nx.get_node_attributes(self.network, 'weights')[node_id]

        # Only going to merge up to the min num of components among neighbors
        min_comp = min([len(node_comps)] +
                       [len(cs) for n, cs in neighbor_comps.items()])

        sum_covs_comp = []
        for i in range(min_comp):
            comp_cov = node_comps[i].state_cov
            sum_covs = comp_cov / weight
            for neighbor, n_comps in neighbor_comps.items():
                n_weight = nx.get_node_attributes(self.network, 'weights')[neighbor]
                n_comp_cov = n_comps[i].state_cov
                sum_covs += n_comp_cov / n_weight
            sum_covs_comp.append(sum_covs)

        return sum_covs_comp

    # For fusing the alphas, first need to calc all state difference combos
    # def state_diffs(self, node_id):


    @staticmethod
    def kappa(cov, weight):
        numer = np.linalg.det(2 * np.pi * np.linalg.inv(cov * weight)) ** 0.5
        denom = np.linalg.det(2 * np.pi * cov) ** (weight * 0.5)
        return numer / denom

















