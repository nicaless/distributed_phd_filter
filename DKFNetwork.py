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

#from optimization_utils import *
#from reconfig_utils import *
if platform.system() == 'Linux':
    print('loading in files with jit')
    from optimization_utils_jit import *
    from reconfig_utils_jit import *
else:
    from optimization_utils import *
    from reconfig_utils import *


class DKFNetwork:
    def __init__(self,
                 nodes,
                 weights,
                 G,
                 targets):
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, weights, 'weights')
        self.targets = targets

        # TRACKERS
        self.failures = {}
        self.adjacencies = {}
        self.weighted_adjacencies = {}
        self.errors = {}  # TODO: figure out real error calculation?
        self.max_trace_cov = {}
        self.mean_trace_cov = {}


    """
    Simulation Operations
    """
    def step_through(self, inputs,
                     L=3, opt='agent',
                     fail_int=None, fail_sequence=None,
                     single_node_fail=False,
                     base=False, noise_mult=1):
        nodes = nx.get_node_attributes(self.network, 'node')

        failure = False
        for i, ins in inputs.items():
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
            True Target Update With Inputs
            """
            # TODO:

            """
            Local Target Estimation
            """
            for id, n in nodes.items():
                n.predict(inputs=ins)
                measurements = n.get_measurements()
                n.update(measurements)
                n.update_trackers(i)


            """
            Do Optimization and Formation Synthesis
            """
            # TODO: fix to remove geom/arithmetic fusion
            # if failure and not base:
            #     if opt == 'agent':
            #         self.do_agent_opt(fail_node)
            #     elif opt == 'team':
            #         self.do_team_opt()
            #     elif opt == 'greedy':
            #         self.do_greedy_opt(fail_node)
            #
            #     # Random strategy
            #     else:
            #         self.do_random_opt(fail_node)
            #
            #     # Formation Synthesis
            #     current_coords = {nid: n.position for nid, n in nodes.items()}
            #     fov = {nid: n.fov for nid, n in nodes.items()}
            #     centroid = self.get_centroid(fail_node)
            #
            #     new_coords = generate_coords(self.adjacency_matrix(),
            #                                  current_coords, fov, centroid)
            #     if new_coords:
            #         for id, n in nodes.items():
            #             n.update_position(new_coords[id])
            #     failure = False

            """
            Consensus
            """
            for id, n in nodes.items():
                n.init_consensus()

            for l in range(L):
                neighbor_weights = {}
                neighbor_omegas = {}
                neighbor_qs = {}

                for id, n in nodes.items():
                    weights = []
                    omegas = []
                    qs = []
                    for neighbor in self.network.neighbors(id):
                        n_weight = nx.get_node_attributes(self.network,
                                                          'weights')[neighbor]
                        n_node = nx.get_node_attributes(self.network,
                                                        'node')[neighbor]
                        weights.append(n_weight)
                        omegas.append(n_node.omega)
                        qs.append(n_node.qs)
                    neighbor_weights[n] = weights
                    neighbor_omegas[n] = omegas
                    neighbor_qs[n] = qs

                for id, n in nodes.items():
                    n.consensus_filter(neighbor_weights[n],
                                       neighbor_omegas[n],
                                       neighbor_qs[n])
            for id, n in nodes.items():
                n.after_consensus_update()

            for id, n in nodes.items():
                n.update_trackers(i, pre_consensus=False)

            trace_covs = self.get_trace_covariances()
            self.max_trace_cov[i] = max(trace_covs)
            self.mean_trace_cov[i] = np.mean(trace_covs)
            self.errors[i] = self.calc_errors(self.targets)
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


    """
    Optimization  
    """
    # TODO: Revisit ICRA Formula
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
