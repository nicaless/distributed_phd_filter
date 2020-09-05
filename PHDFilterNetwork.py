import math
import networkx as nx
from numba import jit, njit, cuda
import numpy as np
from operator import attrgetter
import pandas as pd
import platform
import scipy
from scipy.spatial.distance import mahalanobis

from ospa import *
from target import Target

from optimization_utils import *
from reconfig_utils import *
# if platform.system() == 'Linux':
#     print('loading in files with jit')
#     from optimization_utils_jit import *
#     from reconfig_utils_jit import *
# else:
#     from optimization_utils import *
#     from reconfig_utils import *


class PHDFilterNetwork:
    def __init__(self,
                 nodes,
                 weights,
                 G,
                 merge_thresh=0.2):
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, weights, 'weights')
        self.target_estimates = []

        """
        Dictionary of form {node_id : components} which indicates which 
        components belonging to node_id to share with neighbors 
        """
        self.node_share = {}

        """
        Dictionary of form {node_id : {neighbor_id: components}} which 
        indicates which components of neighbor_id were shared with node_id 
        """
        self.node_neighbor_comps = {}

        """
        Dictionary of form {node_id : {comp_id: components}} which 
        indicates which components to fuse with the comp_id of node_id 
        """
        self.node_fuse_comps = {}

        """
        Dictionary of form {node_id : {comp_id: weights}} which indicates 
        which weights to use when fusing components in self.node_fuse_comps 
        with comp_id of node_id 
        """
        self.node_fuse_weights = {}

        """
        Dictionary of form {node_id : cardinality_estimate} the estimate 
        of total targets for node_id 
        """
        self.cardinality = {}

        self.merge_thresh = merge_thresh

        # TRACKERS
        self.failures = {}
        self.adjacencies = {}
        self.weighted_adjacencies = {}
        self.errors = {}
        self.max_trace_cov = {}
        self.mean_trace_cov = {}
        self.gospa = {}
        self.nmse_card = {}


    """
    Simulation Operations
    """
    def step_through(self, measurements, true_targets,
                     L=3, how='geom', opt='agent',
                     fail_int=None, fail_sequence=None,
                     single_node_fail=False,
                     base=False, noise_mult=1):
        nodes = nx.get_node_attributes(self.network, 'node')

        if not isinstance(measurements, dict):
            measurements = {0: measurements}
            true_targets = {0: true_targets}

        failure = False
        for i, m in measurements.items():
            if fail_int is not None or fail_sequence is not None:
                if fail_int is not None:
                    if i in fail_int:
                        failure = True
                        fail_node = self.apply_failure(i, mult=noise_mult, single_node_fail=single_node_fail)
                else:
                    if i in fail_sequence:
                        failure = True
                        fail_node = self.apply_failure(i, fail=fail_sequence[i], single_node_fail=single_node_fail)

            """
            Local PHD Estimation
            """
            for id, n in nodes.items():
                n.step_through(m, i)

            """
            Do Optimization and Formation Synthesis
            """
            if failure and not base:
                if opt == 'agent':
                    self.do_agent_opt(fail_node, how=how)
                elif opt == 'team':
                    self.do_team_opt(how=how)
                elif opt == 'greedy':
                    self.do_greedy_opt(fail_node, how=how)

                # Random strategy
                else:
                    self.do_random_opt(fail_node)

                # Formation Synthesis
                current_coords = {nid: n.position for nid, n in nodes.items()}
                fov = {nid: n.fov for nid, n in nodes.items()}
                centroid = self.get_centroid(fail_node)

                new_coords = generate_coords(self.adjacency_matrix(),
                                             current_coords, fov, centroid)
                if new_coords:
                    for id, n in nodes.items():
                        n.update_position(new_coords[id])
                failure = False

            """
            Core Fusion Steps
            1) Cardinality Consensus
            2) Geometric or Arithmetic Fusion
            3) Rescaling fused weights according to cardinality consensus
            """
            L = int(max(3.0, len(nodes) / 2.0))
            for l in range(L):
                self.cardinality_consensus()
                fused_comps = self.fuse_components(how=how)
                self.rescale_component_weights()

            for id, n in nodes.items():
                # n.targets = fused_comps[id]
                # print(id, len(n.targets))
                n.update_trackers(i, pre_consensus=False)

            trace_covs = self.get_trace_covariances()
            self.max_trace_cov[i] = max(trace_covs)
            self.mean_trace_cov[i] = np.mean(trace_covs)
            self.errors[i] = self.calc_errors(true_targets[i])
            self.gospa[i] = self.calc_ospa(true_targets[i])
            self.nmse_card[i] = self.calc_nmse_card(true_targets[i])
            self.adjacencies[i] = self.adjacency_matrix()
            self.weighted_adjacencies[i] = self.weighted_adjacency_matrix()

    def apply_failure(self, i, fail=None, mult=1, single_node_fail=False):
        nodes = nx.get_node_attributes(self.network, 'node')

        # Generate new R
        if fail is None:
            if single_node_fail:
                fail_node = 0
            else:
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

    def get_centroid(self, fail_node):
        node = nx.get_node_attributes(self.network, 'node')[fail_node]

        all_phd_states = []
        for t in node.targets:
            pos = np.array([[t.state[0][0]], [t.state[1][0]]])
            all_phd_states.append(pos)

        if len(all_phd_states) == 0:
            return np.array([[0], [0]])

        x, y = zip(*all_phd_states)
        center_x = sum(x) / float(len(x))
        center_y = sum(y) / float(len(x))
        return np.array([[center_x], [center_y]])

    # TODO: remove, probably not needed anymore
    def reduce_comps(self, node_id):
        node = nx.get_node_attributes(self.network, 'node')[node_id]
        node_comps = node.targets
        limit = min(self.cardinality[node_id], node.max_components)
        self.cardinality[node_id] = limit
        keep_node_comps = node_comps[:limit]
        node.targets = keep_node_comps

    """
    Optimization
    """
    def construct_blockdiag_cov(self, node_id, min_cardinality):
        node = nx.get_node_attributes(self.network, 'node')[node_id]
        node_comps = node.targets

        if len(node_comps) == 1:
            return node_comps[0].state_cov
        else:
            bd = scipy.linalg.block_diag(node_comps[0].state_cov,
                                         node_comps[1].state_cov)
            for i in range(2, int(np.floor(min_cardinality))):
                bd = scipy.linalg.block_diag(bd, node_comps[i].state_cov)
            return bd

    def prep_optimization_data(self, how='geom'):
        # get min_cardinality
        min_cardinality_index = min(self.cardinality.keys(),
                                    key=(lambda k: self.cardinality[k]))
        min_cardinality = self.cardinality[min_cardinality_index]

        # get the covariance data
        cov_data = []
        for node_id in list(self.network.nodes()):
            P = self.construct_blockdiag_cov(node_id, min_cardinality)
            if how == 'geom':
                P = np.linalg.inv(P)
            cov_data.append(P)

        return min_cardinality, cov_data

    def do_agent_opt(self, failed_node, how='geom'):
        nodes = nx.get_node_attributes(self.network, 'node')

        min_cardinality, cov_data = self.prep_optimization_data(how=how)

        current_weights = nx.get_node_attributes(self.network, 'weights')

        covariance_data = []
        for c in cov_data:
            det_c = np.linalg.det(c)
            covariance_data.append((1/min_cardinality) * det_c)

        new_config, new_weights = agent_opt(self.adjacency_matrix(),
                                            current_weights,
                                            covariance_data,
                                            failed_node=failed_node)
        print(current_weights)
        print(new_weights)
        G = nx.from_numpy_matrix(new_config)
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, new_weights, 'weights')
        # G = nx.from_numpy_matrix(new_config)
        # self.network = G
        # nx.set_node_attributes(self.network, nodes, 'node')
        #
        # new_weights = self.get_metro_weights()
        # nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_team_opt(self, how='geom'):
        nodes = nx.get_node_attributes(self.network, 'node')

        min_cardinality, cov_data = self.prep_optimization_data(how=how)

        current_weights = nx.get_node_attributes(self.network, 'weights')

        # new_config, new_weights = team_opt(self.adjacency_matrix(),
        #                                     current_weights,
        #                                     cov_data)
        new_config, new_weights = team_opt2(self.adjacency_matrix(),
                                            current_weights,
                                            cov_data,
                                            how=how)
        print(current_weights)
        print(new_weights)
        G = nx.from_numpy_matrix(new_config)
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, new_weights, 'weights')
        # G = nx.from_numpy_matrix(new_config)
        # self.network = G
        # nx.set_node_attributes(self.network, nodes, 'node')
        #
        # new_weights = self.get_metro_weights()
        # nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_greedy_opt(self, failed_node, how='geom'):
        nodes = nx.get_node_attributes(self.network, 'node')

        min_cardinality, cov_data = self.prep_optimization_data(how=how)

        current_neighbors = list(self.network.neighbors(failed_node))

        best_cov_id = None
        best_cov = np.inf
        for neighbor_id in list(nodes):
            if neighbor_id not in current_neighbors:
                if np.linalg.det(cov_data[neighbor_id]) < best_cov:
                    best_cov_id = neighbor_id

        if best_cov_id is None:
            pass
        else:
            new_config = self.adjacency_matrix()
            new_config[failed_node, best_cov_id] = 1
            new_config[best_cov_id, failed_node] = 1

            G = nx.from_numpy_matrix(new_config)
            self.network = G
            nx.set_node_attributes(self.network, nodes, 'node')

            new_weights = self.get_metro_weights()
            nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_random_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')
        current_neighbors = list(self.network.neighbors(failed_node))

        non_neighbors = []
        for neighbor_id in list(nodes):
            if neighbor_id not in current_neighbors:
                non_neighbors.append(neighbor_id)

        if len(non_neighbors) == 0:
            pass
        else:
            new_neighbor_id = np.random.choice(non_neighbors)

            new_config = self.adjacency_matrix()
            new_config[failed_node, new_neighbor_id] = 1
            new_config[new_neighbor_id, failed_node] = 1

            G = nx.from_numpy_matrix(new_config)
            self.network = G
            nx.set_node_attributes(self.network, nodes, 'node')

            new_weights = self.get_metro_weights()
            nx.set_node_attributes(self.network, new_weights, 'weights')


    """
    Core Fusion Steps
    1) Cardinality Consensus
    2) Geometric or Arithmetic Fusion
    3) Rescaling fused weights according to cardinality consensus
    """

    def cardinality_consensus(self):
        """
        1) Each sensor gets local estimation of number of targets
        2a) Each sensor shares local estimation with neighbors
        2b) Each sensor updates their local estimation of number of targets

        Note: the third step weight scaling occurs after the
        geometric/arithmetic fusion

        :return:
        """
        nodes = nx.get_node_attributes(self.network, 'node')
        metro_weights = nx.get_node_attributes(self.network, 'weights')

        """
        Get Local Estimation of number of targets
        """
        local_estimates = {}
        for n in list(self.network.nodes()):
            all_weights = [t.weight for t in nodes[n].targets]
            sum_weights = np.nansum(all_weights)
            tot_targets = len(all_weights)
            local_estimates[n] = min([tot_targets, sum_weights])

        """
        Share Local Estimation with neighbors
        
        shared_estimates is dict of form: {node_id: {neighbor_id: estimate}}
        """
        shared_estimates = {}

        for node_id in list(self.network.nodes()):
            neighbors = list(self.network.neighbors(node_id))
            neighbor_estimates = {}
            for neighbor_id in neighbors:
                neighbor_estimates[neighbor_id] = local_estimates[neighbor_id]
            shared_estimates[node_id] = neighbor_estimates

        """
        Updates local estimation of number of targets via fusion with 
        neighbor estimates 
        """
        cardinality = {}
        for node_id in list(self.network.nodes()):
            weighted_estimate = 0

            neighbor_weights = metro_weights[node_id]
            for neighbor_id, estimate in shared_estimates[node_id].items():
                neighbor_estimate = estimate
                neighbor_weight = neighbor_weights[neighbor_id]
                weighted_estimate += neighbor_weight * neighbor_estimate
            # TODO: is ceil the right way to go?
            if np.floor(weighted_estimate) > len(nodes[node_id].targets):
                c = len(nodes[node_id].targets)
            else:
                c = np.floor(weighted_estimate)
            cardinality[node_id] = c
        self.cardinality = cardinality

    def fuse_components(self, how='geom'):

        """
        Run utils to help with fusion
        """
        self.share_comps()
        self.get_neighbors_comps()
        self.get_comps_to_fuse()

        new_comps = {}

        for node_id in list(self.network.nodes()):
            if how == 'geom':
                fused_comps = self.geometric_fusion(node_id)
            else:
                fused_comps = self.arithmetic_fusion(node_id)
            new_comps[node_id] = fused_comps
        return new_comps

    def rescale_component_weights(self):
        nodes = nx.get_node_attributes(self.network, 'node')

        for node_id in list(self.network.nodes()):
            old_estimate = np.nansum([t.weight
                                      for t in nodes[node_id].targets])
            new_estimate = self.cardinality[node_id]

            if old_estimate == 0:
                rescaler = 1
            else:
                rescaler = new_estimate / old_estimate

            targets = nodes[node_id].targets
            for t in targets:
                t.weight = rescaler * t.weight
            targets.sort(key=attrgetter('weight'), reverse=True)
            nodes[node_id].targets = targets

    """
    Fusion Utils (for both Geometric and Arithmetic fusion)
    """

    def share_comps(self):
        """
        Prerequisite: after local PHD estimates

        Creates dictionary of form {node_id : components} which indicates which
        components belonging to node_id to share with neighbors

        :param node_id:
        :return:
        """
        node_share = {}

        for node_id in list(self.network.nodes()):
            node = nx.get_node_attributes(self.network, 'node')[node_id]
            node_share[node_id] = node.targets

        self.node_share = node_share

    def get_neighbors_comps(self):
        """
        Prerequisite: after local PHD estimates

        :return: None
        Dictionary of form {node_id : {neighbor_id: components}} which
        indicates which components of neihbor_id were shared with node_id
        """
        G = self.network

        node_neighbor_comps = {}
        for node_id in list(self.network.nodes()):
            node_neighbor_comps[node_id] = {}

            neighbors = list(G.neighbors(node_id))
            for neighbor_id in neighbors:
                neighbor_node = nx.get_node_attributes(G, 'node')[neighbor_id]
                neighbor_comps = neighbor_node.targets
                node_neighbor_comps[node_id][neighbor_id] = neighbor_comps

        self.node_neighbor_comps = node_neighbor_comps

    def get_comps_to_fuse(self):
        """

        Prerequisite: after sharing comps and getting neighbor comps

        Updates the dictionaries self.node_fuse_comps and
        self.node_fuse_weights to aid fusion

        :param node_id:
        :return None:
        """
        G = self.network
        for node_id in list(self.network.nodes()):
            node_comps = self.node_share[node_id]
            neighbor_comps = self.node_neighbor_comps[node_id]

            metro_weights = nx.get_node_attributes(self.network,
                                                   'weights')[node_id]

            node_weight = metro_weights[node_id]

            self.node_fuse_comps[node_id] = {}
            self.node_fuse_weights[node_id] = {}
            for i in range(len(node_comps)):
                c0 = node_comps[i]

                fuse_comps = [c0]
                fuse_weights = [node_weight]
                for neighbor_id, comps in neighbor_comps.items():
                    """
                    For each neighbor, find closest comp to node_comp[i]
                    within merge_thresh (using mahalanobis distance)
                    """
                    closest_comp = None
                    distances = [self.get_mahalanobis(c1, c0) for c1 in comps]
                    closest_comp_index = int(np.argmin(distances))

                    if distances[closest_comp_index] <= self.merge_thresh:
                        closest_comp = comps[closest_comp_index]

                    if closest_comp is not None:
                        fuse_comps.append(closest_comp)
                        fuse_weights.append(metro_weights[neighbor_id])

                self.node_fuse_comps[node_id][i] = fuse_comps
                self.node_fuse_weights[node_id][i] = fuse_weights

    """
    Arithmetic Fusion
    """
    def arithmetic_fusion(self, node_id):
        """
        PRE-REQUISITE: that you ran self.get_comps_to_fuse

        :param node_id:
        :return list of new components for node_i:
        """

        node_comps = self.node_share[node_id]

        covs = self.fuse_covs_arith(node_id)
        states, alphas = self.fuse_states_alphas_arith(node_id)

        fuse_comps = []
        for i in range(len(node_comps)):
            new_comp = Target(init_weight=alphas[i],
                              init_state=states[i],
                              init_cov=covs[i])
            fuse_comps.append(new_comp)
        return fuse_comps

    def fuse_covs_arith(self, node_id):
        node_comps = self.node_share[node_id]

        new_covs = []
        for i in range(len(node_comps)):
            fuse_comps = self.node_fuse_comps[node_id][i]
            fuse_weights = self.node_fuse_weights[node_id][i]

            if len(fuse_comps) == 1:
                """
                If nothing to fuse with component (no close enough 
                components from neighbors to fuse) keep current covariance 
                """
                new_covs.append(node_comps[i].state_cov)
            else:
                """
                Rescale weights to equal 1 if necessary
                (only necessary if not all neighbors contributed)
                """
                sum_fuse_weights = float(sum(fuse_weights))
                fuse_weights = [fw / sum_fuse_weights for fw in fuse_weights]

                """
                Fuse According to Arith Fusion in paper
                """
                fuse_comps_weighted = []
                for j in range(len(fuse_comps)):
                    w = fuse_weights[j]
                    cov = fuse_comps[j].state_cov
                    fuse_comps_weighted.append(w * cov)
                sum_covs = np.sum(fuse_comps_weighted, 0)
                new_covs.append(sum_covs)

        return new_covs

    def fuse_states_alphas_arith(self, node_id):
        node_comps = self.node_share[node_id]

        new_states = []
        new_alphas = []
        for i in range(len(node_comps)):
            fuse_comps = self.node_fuse_comps[node_id][i]

            if len(fuse_comps) == 1:
                """
                If nothing to fuse with component (no close enough 
                components from neighbors to fuse) keep current state 
                """
                new_states.append(node_comps[i].state)
                new_alphas.append(node_comps[i].weight)
            else:
                """
                Fuse According to Arith Fusion in paper
                """
                fuse_alpha_weighted_states = []
                fuse_alphas = []

                for j in range(len(fuse_comps)):
                    alpha = fuse_comps[j].weight
                    x = fuse_comps[j].state
                    fuse_alpha_weighted_states.append(alpha * x)
                    fuse_alphas.append(alpha)
                sum_alpha_weighted_states = np.sum(fuse_alpha_weighted_states,
                                                   0)
                sum_alphas = np.sum(fuse_alphas)

                if sum_alphas == 0:
                    new_states.append(sum_alpha_weighted_states)
                    new_alphas.append(sum_alphas)
                    continue

                new_states.append(sum_alpha_weighted_states / sum_alphas)
                new_alphas.append(sum_alphas)

        return new_states, new_alphas


    """
    Geometric Fusion
    """

    def geometric_fusion(self, node_id):
        """
        PRE-REQUISITE: that you ran self.get_comps_to_fuse

        :param node_id:
        :return list of new components for node_i:
        """

        node_comps = self.node_share[node_id]

        Ks = self.get_k(node_id)

        covs = self.fuse_covs_geom(node_id)
        states = self.fuse_states_geom(node_id, covs)
        alphas = self.fuse_alphas_geom(node_id, Ks)

        fuse_comps = []
        for i in range(len(node_comps)):
            new_comp = Target(init_weight=alphas[i],
                              init_state=states[i],
                              init_cov=covs[i])
            fuse_comps.append(new_comp)
        return fuse_comps

    def get_k(self, node_id):
        """
        For each node_id, for each set of components to fuse, calculate K factor
        :param node_id:
        :return list of Ks:
        """

        node_comps = self.node_share[node_id]

        Ks = []
        for i in range(len(node_comps)):
            fuse_comps = self.node_fuse_comps[node_id][i]
            fuse_weights = self.node_fuse_weights[node_id][i]

            if len(fuse_comps) == 1:
                """
                If nothing to fuse with component (no close enough 
                components from neighbors to fuse) no need to calculate K 
                """
                Ks.append(1)
            else:
                """
                Rescale weights to equal 1 if necessary
                (only necessary if not all neighbors contributed)
                """
                sum_fuse_weights = float(sum(fuse_weights))
                fuse_weights = [fw / sum_fuse_weights for fw in fuse_weights]

                k = self.calcK(fuse_comps, fuse_weights)
                Ks.append(k)

        return Ks

    def fuse_covs_geom(self, node_id):
        """
        For each component of node_id, fuse with the closest components from
        neighboprs of node_id

        :param node_id:
        :return new covariances:
        """

        node_comps = self.node_share[node_id]

        new_covs = []
        for i in range(len(node_comps)):
            fuse_comps = self.node_fuse_comps[node_id][i]
            fuse_weights = self.node_fuse_weights[node_id][i]

            if len(fuse_comps) == 1:
                """
                If nothing to fuse with component (no close enough 
                components from neighbors to fuse) keep current covariance 
                """
                new_covs.append(node_comps[i].state_cov)
            else:
                """
                Rescale weights to equal 1 if necessary
                (only necessary if not all neighbors contributed)
                """
                sum_fuse_weights = float(sum(fuse_weights))
                fuse_weights = [fw / sum_fuse_weights for fw in fuse_weights]

                """
                Fuse According to Geom Fusion in paper
                """
                fuse_comps_weighted = []
                for j in range(len(fuse_comps)):
                    w = fuse_weights[j]
                    inv_cov = np.linalg.inv(fuse_comps[j].state_cov)
                    fuse_comps_weighted.append(w * inv_cov)
                sum_covs = np.sum(fuse_comps_weighted, 0)
                new_covs.append(np.linalg.inv(sum_covs))

        return new_covs

    def fuse_states_geom(self, node_id, new_covs):
        """
        For each component of node_id, fuse with the closest components from
        neighboprs of node_id

        :param node_id:
        :return new states:
        """

        node_comps = self.node_share[node_id]

        new_states = []
        for i in range(len(node_comps)):
            fuse_comps = self.node_fuse_comps[node_id][i]
            fuse_weights = self.node_fuse_weights[node_id][i]

            if len(fuse_comps) == 1:
                """
                If nothing to fuse with component (no close enough 
                components from neighbors to fuse) keep current state 
                """
                new_states.append(node_comps[i].state)
            else:
                """
                Rescale weights to equal 1 if necessary
                (only necessary if not all neighbors contributed)
                """
                sum_fuse_weights = float(sum(fuse_weights))
                fuse_weights = [fw / sum_fuse_weights for fw in fuse_weights]

                """
                Fuse According to Geom Fusion in paper
                """
                fuse_comps_weighted = []
                for j in range(len(fuse_comps)):
                    w = fuse_weights[j]
                    inv_cov = np.linalg.inv(fuse_comps[j].state_cov)
                    x = fuse_comps[j].state
                    fuse_comps_weighted.append(w * inv_cov * x)
                sum_states = np.sum(fuse_comps_weighted, 0)
                new_states.append(new_covs[i] * sum_states)

        return new_states

    def fuse_alphas_geom(self, node_id, K):
        """
        For each component of node_id, fuse with the closest components from
        neighboprs of node_id

        :param node_id:
        :return new alphas:
        """

        node_comps = self.node_share[node_id]

        new_alphas = []
        for i in range(len(node_comps)):
            fuse_comps = self.node_fuse_comps[node_id][i]
            fuse_weights = self.node_fuse_weights[node_id][i]

            if len(fuse_comps) == 1:
                """
                If nothing to fuse with component (no close enough 
                components from neighbors to fuse) keep current alpha 
                (component weight) 
                """
                new_alphas.append(node_comps[i].weight)
            else:
                """
                Rescale weights to equal 1 if necessary
                (only necessary if not all neighbors contributed)
                """
                sum_fuse_weights = float(sum(fuse_weights))
                fuse_weights = [fw / sum_fuse_weights for fw in fuse_weights]

                """
                Fuse According to Geom Fusion in paper
                """
                fuse_comps_weighted = []
                for j in range(len(fuse_comps)):
                    w = fuse_weights[j]
                    alpha = fuse_comps[j].weight
                    alpha_w = alpha ** w
                    r = self.rescaler(fuse_comps[j].state_cov, fuse_weights[j])
                    fuse_comps_weighted.append(alpha_w * r)
                prod_alpha = np.prod(fuse_comps_weighted, 0)
                new_alphas.append(prod_alpha * K[i])

        return new_alphas


    """
    Network Operations
    """

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

    def get_metro_weights(self):
        G = self.network
        num_nodes = len(list(self.network.nodes))
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

        return weight_attrs


    """
    Metric Calculations
    """

    def get_trace_covariances(self):
        min_cardinality_index = min(self.cardinality.keys(),
                                    key=(lambda k: self.cardinality[k]))
        min_cardinality = self.cardinality[min_cardinality_index]

        cov_data = []
        for node_id in list(self.network.nodes()):
            P = self.construct_blockdiag_cov(node_id, min_cardinality)
            cov_data.append(np.trace(P))

        return cov_data

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

        if N == 0 and sum_card == 0:
            return 0

        if N == 0 or sum_card == 0:
            return squared_error

        return squared_error / (N * sum_card)


    """
    Saving Data
    """

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

    """
    Static Methods
    """

    @staticmethod
    def calcK(comps, weights):
        # Omega and q
        covs_weighted = []
        states_weighted = []
        for j in range(len(comps)):
            w = weights[j]
            inv_cov = np.linalg.inv(comps[j].state_cov)
            x = comps[j].state
            covs_weighted.append(w * inv_cov)
            states_weighted.append(w * np.dot(inv_cov, x))
        omega = np.sum(covs_weighted, 0)
        q = np.sum(states_weighted, 0)

        d = comps[0].state.shape[0]

        # Kappa 1:n
        firstterm_1n = len(comps) * d * np.log(2 * np.pi)
        secondterm_1n = 0  # Fill in later
        thirdterm_1n = 0  # Fill in later

        # Kappa n
        firstterm_n = d * np.log(2 * np.pi)
        secondterm_n = 0  # Fill in later
        thirdterm_n = np.dot(q.T, np.dot(np.linalg.inv(omega), q))

        for i in range(len(comps)):
            weight = weights[i]
            state = comps[i].state
            cov = comps[i].state_cov

            secondterm_1n += np.log(np.linalg.det(weight * np.linalg.inv(cov)))
            thirdterm_1n += weight * np.dot(state.T, np.dot(np.linalg.inv(cov),
                                                            state))

            secondterm_n += np.linalg.det(weight * np.linalg.inv(cov))

        kappa_1n = -0.5 * (firstterm_1n - secondterm_1n + thirdterm_1n)
        kappa_n = -0.5 * (firstterm_n - np.log(secondterm_n) + thirdterm_n)

        K = np.exp(kappa_1n[0][0] - kappa_n[0][0])
        return K

    @staticmethod
    @njit
    def rescaler(cov, weight):
        numer = np.linalg.det(2 * np.pi * np.linalg.inv(cov / weight))
        denom = np.linalg.det(2 * np.pi * cov) ** weight
        return (numer / denom) ** 0.5

    @staticmethod
    def get_mahalanobis(target1, target2):
        d = mahalanobis(target1.state, target2.state,
                        np.linalg.inv(target1.state_cov))
        return d
