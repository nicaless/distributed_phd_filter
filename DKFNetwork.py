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

from optimization_utils_dkf import *
from reconfig_utils_dkf import *


class DKFNetwork:
    def __init__(self,
                 nodes,
                 weights,
                 G,
                 targets):
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, weights, 'weights')
        self.targets = targets  # the true targets

        # TRACKERS
        self.failures = {}
        self.adjacencies = {}
        self.weighted_adjacencies = {}
        self.errors = {}
        self.max_trace_cov = {}
        self.mean_trace_cov = {}
        self.surveillance_quality = {}


    """
    Simulation Operations
    """
    # TODO: measurements param not needed
    def step_through(self, inputs, measurements=None,
                     opt='agent', L=10, fail_int=None, fail_sequence=None,
                     single_node_fail=False,
                     base=False, noise_mult=1):
        nodes = nx.get_node_attributes(self.network, 'node')

        failure = False
        for i, ins in inputs.items():
            print(i)
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
            for t, target in enumerate(self.targets):
                # TODO check target next state before actually doing next state
                # and divert target in negative course if about to oob
                target.next_state(input=ins[t])

            """
            Local Target Estimation
            """
            for id, n in nodes.items():
                n.predict(len(nodes))
                ms = n.get_measurements(self.targets)
                if failure:
                    ms = [m + np.random.random(m.shape) * noise_mult for m in ms]
                n.update(ms)
                n.update_trackers(i)

            """
            Init Consensus
            """
            for id, n in nodes.items():
                n.init_consensus()

            """
            Do Optimization and Formation Synthesis
            """
            if failure and not base:
                if opt == 'agent':
                    self.do_agent_opt(fail_node)
                elif opt == 'team':
                    self.do_team_opt()
                elif opt == 'greedy':
                    self.do_greedy_opt(fail_node)

                # Random strategy
                else:
                    self.do_random_opt(fail_node)

                # Formation Synthesis
                current_coords = {nid: n.position for nid, n in nodes.items()}
                fov = {nid: n.fov for nid, n in nodes.items()}
                Rs = {nid: n.R for nid, n in nodes.items()}

                new_coords, sq = generate_coords(self.adjacency_matrix(),
                                                 current_coords, fov, Rs)
                self.surveillance_quality[i] = sq
                if new_coords:
                    for id, n in nodes.items():
                        n.update_position(new_coords[id])
                failure = False

            """
            Run Consensus
            """
            for l in range(L):
                neighbor_weights = {}
                neighbor_omegas = {}
                neighbor_qs = {}

                for id, n in nodes.items():
                    weights = []
                    omegas = []
                    qs = []
                    n_weights = nx.get_node_attributes(self.network,
                                                      'weights')[id]
                    for neighbor in self.network.neighbors(id):
                        n_node = nx.get_node_attributes(self.network,
                                                        'node')[neighbor]
                        weights.append(n_weights[neighbor])
                        omegas.append(n_node.omega)
                        qs.append(n_node.qs)
                    neighbor_weights[id] = weights
                    neighbor_omegas[id] = omegas
                    neighbor_qs[id] = qs

                for id, n in nodes.items():
                    n.consensus_filter(neighbor_omegas[id],
                                       neighbor_qs[id],
                                       neighbor_weights[id])

            for id, n in nodes.items():
                n.intermediate_cov_update()

            """
            After Consensus Update
            """
            for id, n in nodes.items():
                n.after_consensus_update(len(nodes))

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
    def do_agent_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')
        current_weights = nx.get_node_attributes(self.network, 'weights')

        # cov_data = [n.full_cov for id, n in nodes.items()]
        # cov_data = [n.intermediate_cov for id, n in nodes.items()]
        cov_data = [n.omega for id, n in nodes.items()]

        covariance_data = []
        for c in cov_data:
            trace_c = np.trace(c)
            covariance_data.append(trace_c)

        new_config, new_weights = agent_opt(self.adjacency_matrix(),
                                            current_weights,
                                            covariance_data,
                                            failed_node=failed_node)

        G = nx.from_numpy_matrix(new_config)
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_team_opt(self):
        nodes = nx.get_node_attributes(self.network, 'node')
        current_weights = nx.get_node_attributes(self.network, 'weights')

        # cov_data = [n.full_cov for id, n in nodes.items()]
        # cov_data = [n.intermediate_cov for id, n in nodes.items()]
        cov_data = [n.full_cov_prediction for id, n in nodes.items()]
        omega_data = [n.omega for id, n in nodes.items()]

        new_config, new_weights = team_opt2(self.adjacency_matrix(),
                                            current_weights,
                                            cov_data,
                                            omega_data)
        G = nx.from_numpy_matrix(new_config)
        self.network = G
        nx.set_node_attributes(self.network, nodes, 'node')
        nx.set_node_attributes(self.network, new_weights, 'weights')

    def do_greedy_opt(self, failed_node):
        nodes = nx.get_node_attributes(self.network, 'node')

        current_neighbors = list(self.network.neighbors(failed_node))

        # cov_data = [n.full_cov for id, n in nodes.items()]
        # cov_data = [n.intermediate_cov for id, n in nodes.items()]
        cov_data = [n.full_cov_prediction for id, n in nodes.items()]

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
        nodes = nx.get_node_attributes(self.network, 'node')

        cov_data = []
        for id, node in nodes.items():
            cov_data.append(np.trace(node.full_cov))

        return cov_data

    # TODO:
    def calc_errors(self, true_targets):
        nodes = nx.get_node_attributes(self.network, 'node')

        node_errors = {}
        for id, node in nodes.items():
            errors = []
            for i, t in enumerate(true_targets):
                # e = mahalanobis(node.targets[i].state,
                #                 t.state,
                #                 node.targets[i].state_cov)
                e = np.linalg.norm(node.targets[i].state - t.state)
                errors.append(e)
            node_errors[id] = errors

        return node_errors


    """
    Saving Data
    """
    def save_metrics(self, path):
        # Save Errors
        errors = pd.DataFrame.from_dict(self.errors, orient='index')
        for i in range(len(self.network.nodes)):
            errors[i] = errors[i].apply(np.nanmean)
        errors.to_csv(path + '/errors.csv', index_label='time')

        # Save Max Trace Cov
        max_tr_cov = pd.DataFrame.from_dict(self.max_trace_cov, orient='index')
        max_tr_cov.columns = ['value']
        max_tr_cov.to_csv(path + '/max_tr_cov.csv', index_label='time')

        # Save Mean Trace Cov
        mean_tr_cov = pd.DataFrame.from_dict(self.mean_trace_cov, orient='index')
        mean_tr_cov.columns = ['value']
        mean_tr_cov.to_csv(path + '/mean_tr_cov.csv', index_label='time')

        # Save Surveillance Quality
        if len(self.surveillance_quality) != 0:
            surv_q = pd.DataFrame.from_dict(self.surveillance_quality, orient='index')
            surv_q.columns = ['value']
            surv_q.to_csv(path + '/surveillance_quality.csv', index_label='time')

    def save_estimates(self, path):
        all_nodes = nx.get_node_attributes(self.network, 'node')
        time = []
        x = []
        y = []
        ids = []
        for n, node in all_nodes.items():
            for t, pos in node.consensus_positions.items():
                for id, p in enumerate(pos):
                    time.append(t)
                    x.append(p[0][0])
                    y.append(p[1][0])
                    ids.append(id)
        data = pd.DataFrame([time, x, y, ids])
        data = data.transpose()
        data.columns = ['time', 'x', 'y', 'target']
        data.to_csv(path + '/estimates.csv', index=False)

    def save_true_target_states(self, path):
        for i, t in enumerate(self.targets):
            x = []
            y = []
            for state in t.all_states:
                x.append(state[0][0])
                y.append(state[1][0])
            data = pd.DataFrame([x, y])
            data = data.transpose()
            data.columns = ['x', 'y']
            data.to_csv(path + '/target_{t}_positions.csv'.format(t=i),
                        index=False)

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
