from copy import deepcopy
import csv
import cvxopt as cvx
import math
import numpy as np
import os
import picos as pic
from picos.solvers import SolverError
import platform
from scipy.linalg import block_diag
from heapq import *
import itertools
counter = itertools.count()


def magnitude(x):
    return int(math.log10(abs(x)))


def agent_opt(adj_mat, current_weights, covariance_data, ne=1, failed_node=None):
    """
    Runs the agent optimization problem

    :param adj_mat: the Adjacency matrix
    :param current_weights: current node weights
    :param covariance_data: list of each node's covariance matrix
    :param ne: limit for number of edges to change
    :param failed_node: the node that fails
    :return: new adjacency matrix and new weights
    """
    edge_mod_limit = ne * 2
    n = adj_mat.shape[0]
    beta = 1 / n
    node_bin = np.zeros((1, n))
    if failed_node is not None:
        node_bin[0][failed_node] = 1

    # Reducing Magnitude if necessary
    covariance_data = np.nan_to_num(covariance_data)
    magnitude_covs = [magnitude(cov) for cov in covariance_data]
    if max(magnitude_covs) > 15:
        print('rescaling matrix magnitude, magnitude too high')
        covariance_data = [cov * (10 ** (-1 * max(magnitude_covs)))
                           for cov in covariance_data]

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape,
                             'symmetric')  # A is an SDP var
    PI = problem.add_variable('PI', adj_mat.shape,
                              'binary')  # PI is a binary var
    mu = problem.add_variable('mu', 1)  # mu is an SDP var

    # Set Objective
    problem.set_objective('min',
                          -1 * node_bin * A * np.array(covariance_data).T)

    # Set Constraints
    problem.add_constraint(mu >= 0.001)
    problem.add_constraint(mu < 1)
    problem.add_constraint(
        (A * np.ones((n, 1))) == np.ones((n, 1)))  # Constraint 1
    problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                           (1 - mu) * np.eye(n) >= A)  # Constraint 2

    for i in range(n):
        problem.add_constraint(A[i, i] > 0)  # Constraint 6
        for j in range(n):
            if i == j:
                problem.add_constraint(PI[i, j] == 1.0)  # Constraint 3
            else:
                problem.add_constraint(A[i, j] > 0)  # Constraint 7
                problem.add_constraint(A[i, j] <= PI[i, j])  # Constraint 8

    problem.add_constraint(
        abs(PI - adj_mat) ** 2 <= edge_mod_limit)  # Constraint 9

    problem.solve(verbose=0, solver='mosek')
    problem_status = problem.status
    print('status: {s}'.format(s=problem_status))
    if problem_status not in ['integer optimal', 'optimal']:
        return adj_mat, current_weights

    new_config = np.zeros(adj_mat.shape)
    new_weights = {}
    for i in range(n):
        new_weights[i] = {}

    for i in range(n):
        nw = A[i, i].value
        if nw == 0:
            nw = 0.1
        new_weights[i][i] = nw
        new_config[i, i] = 1
        for j in range(i + 1, n):
            if round(PI[i, j].value) == 1:
                new_config[i, j] = round(PI[i, j].value)
                new_config[j, i] = round(PI[j, i].value)
                nw = A[i, j].value
                if nw == 0:
                    nw = 0.1
                new_weights[i][j] = nw
                new_weights[j][i] = nw
    print(new_config)
    return new_config, new_weights


def team_opt_matlab(adj_mat, current_weights, covariance_matrices, omegas, ne=3):
    A = adj_mat
    n = adj_mat.shape[0]
    for c, cov in enumerate(covariance_matrices):
        f_name = 'misdp_data/inverse_covariance_matrices/{c}.csv'
        np.savetxt(f_name.format(c=c), np.asarray(np.linalg.inv(cov)), delimiter=",")

    for o, om in enumerate(omegas):
        f_name = 'misdp_data/omega_matrices/{o}.csv'
        np.savetxt(f_name.format(o=o), np.asarray(om), delimiter=",")

    np.savetxt('misdp_data/adj_mat.csv', A, delimiter=",")
    if platform.system() == 'Linux':
        mat_command = 'matlab '
    else:
        mat_command = '/Applications/MATLAB_R2019a.app/bin/matlab '
    matlab_string = mat_command + '-nodesktop -nosplash -r "MISDP_new_copy({e});exit;"'.format(e=ne)
    print('Starting Up MATLAB')
    os.system(matlab_string)
    print('Reading MATLAB results')

    if os.path.exists('misdp_data/new_weights.csv'):
        new_weights = []
        with open('misdp_data/new_weights.csv', 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
            for row in readCSV:
                data = list(map(float, row))
                new_weights.append(data)
        new_weights_mat = np.array(new_weights)
    else:
        new_weights = current_weights

    if os.path.exists('misdp_data/new_A.csv'):
        new_A = []
        with open('misdp_data/new_A.csv', 'r') as f:
            readCSV = csv.reader(f, delimiter=',')
            for row in readCSV:
                data = list(map(float, row))
                new_A.append(data)
        new_A = np.array(new_A)
    else:
        new_A = A

    if 'new_weights_mat' in locals():
        if np.array_equal(new_weights_mat, new_A):
            new_weights = current_weights
        else:
            new_weights = {}
            for i in range(n):
                new_weights[i] = {}
            for i in range(n):
                new_weights[i][i] = new_weights_mat[i, i]
                for j in range(i + 1, n):
                    new_weights[i][j] = new_weights_mat[i, j]
                    new_weights[j][i] = new_weights_mat[j, i]

    print(new_A)
    return new_A, new_weights


def team_opt(adj_mat, current_weights, covariance_matrices, omegas, ne=1):
    """
   Runs the team optimization problem

   :param adj_mat: the Adjacency matrix
   :param current_weights: current node weights
   :param covariance_data: list of each node's large covariance matrix
   :param omegas: list of each node's omega matrix
   :param ne: limit for number of edges to change
   :return: new adjacency matrix and new weights
   """
    edge_mod_limit = ne * 2
    n = adj_mat.shape[0]
    beta = 1 / n
    s = covariance_matrices[0].shape[0]
    # tol = 0.00001
    tol = 0.1
    p_size = n * s

    # Reducing Magnitude if necessary
    mag_max_val_in_matrices = magnitude(max([np.max(abs(cov))
                                             for cov in covariance_matrices]))
    if mag_max_val_in_matrices > 15:
        print('rescaling matrix magnitude, magnitude too high')
        covariance_matrices = [cov * (10 ** (-1 * mag_max_val_in_matrices))
                               for cov in covariance_matrices]

    cov_array = np.zeros((n * s, s))
    for i in range(n):
        start = i * s
        end = i * s + s
        cov_array[start:end, 0:s] = omegas[i]

    inv_cov_array = np.zeros((n * s, s))
    for i in range(n):
        start = i * s
        end = i * s + s
        inv_cov_array[start:end, 0:s] = np.linalg.inv(covariance_matrices[i])

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape,
                             'symmetric')  # A is an SDP var
    PI = problem.add_variable('PI', adj_mat.shape,
                              'binary')  # PI is a binary var
    mu = problem.add_variable('mu', 1)  # mu is an SDP var

    Pbar = problem.add_variable('Pbar', (p_size, p_size))

    delta_list = []
    for i in range(n):
        delta_list.append(problem.add_variable('delta[{0}]'.format(i), (s, s)))

    delta_bar = problem.add_variable('delta_bar', (p_size, p_size))
    delta_array = problem.add_variable('delta_array', (p_size, s))
    schur = problem.add_variable('schur', (p_size * 2, p_size * 2))

    # Add Params (ie constant affine expressions to help with creating constraints)
    I = pic.new_param('I', np.eye(s))
    Ibar = pic.new_param('Ibar', np.eye(p_size))
    cov_array_param = pic.new_param('covs', cov_array)
    inv_cov_array_param = pic.new_param('inv_covs', inv_cov_array)

    # Set Objective
    problem.set_objective('min', beta * pic.trace(Pbar))

    # Constraints

    # Setting Additional Constraint such that delta_bar elements equal elements in delta_list (with some tolerance)
    for i in range(n):
        start = i * s
        end = i * s + s
        problem.add_constraint(abs(delta_bar[start:end, start:end] - delta_list[i]) <= tol)
        if i < (n - 1):
            # Fill everything to left with 0s
            problem.add_constraint(delta_bar[start:end, end:] == np.zeros((s, (n * s) - end)))
            # Fill everything below with 0s
            problem.add_constraint(delta_bar[end:, start:end] == np.zeros(((n * s) - end, s)))

    # Setting Additional Constraint such that delta_array elements equal elements in delta_list (with some tolerance)
    for i in range(n):
        start = i * s
        end = i * s + s
        problem.add_constraint(abs(delta_array[start:end, :] - delta_list[i]) <= tol)

    # Setting Additional Constraint such that delta_bar and Pbar elements in schur variable (with some tolerance)
    problem.add_constraint(abs(schur[0:p_size, 0:p_size] - Pbar) <= tol)
    problem.add_constraint(abs(schur[p_size:, p_size:] - delta_bar) <= tol)
    problem.add_constraint(schur[0:p_size, p_size:] == np.eye(p_size))
    problem.add_constraint(schur[p_size:, 0:p_size] == np.eye(p_size))

    # Schur constraint
    problem.add_constraint(schur >= 0)
    # problem.add_constraint(((Pbar & Ibar) //
    #                        (Ibar & delta_bar)) >> 0)

    # Kron constraint
    problem.add_constraint(pic.kron(A, I) * cov_array_param +
                           inv_cov_array_param == delta_array)
    # problem.add_constraint(pic.kron(A, I) * cov_array_param == delta_array)

    # Set the usual Constraints
    problem.add_constraint(mu >= 0.001)
    problem.add_constraint(mu < 1)
    problem.add_constraint(
        (A * np.ones((n, 1))) == np.ones((n, 1)))  # Constraint 1
    problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                           (1 - mu) * np.eye(n) >= A)  # Constraint 2

    for i in range(n):
        problem.add_constraint(A[i, i] > 0)  # Constraint 6
        for j in range(n):
            if i == j:
                problem.add_constraint(PI[i, j] == 1.0)  # Constraint 3
            else:
                problem.add_constraint(A[i, j] > 0)  # Constraint 7
                problem.add_constraint(A[i, j] <= PI[i, j])  # Constraint 8

    problem.add_constraint(
        abs(PI - adj_mat) ** 2 <= edge_mod_limit)  # Constraint 9

    sol = problem.solve(verbose=0, solver='mosek')
    obj = sol['obj']
    problem_status = problem.status
    print('status: {s}'.format(s=problem_status))
    if problem_status not in ['integer optimal', 'optimal']:
        return adj_mat, current_weights

    # print(np.linalg.inv(delta_bar.value) - Pbar.value)
    # print(np.dot(delta_bar.value, Pbar.value))
    print(np.linalg.det(np.dot(delta_bar.value, Pbar.value)))
    # print(np.trace(np.dot(delta_bar.value, Pbar.value)))
    # print((Pbar | delta_bar).value)
    # print(abs(delta_bar[start:end, start:end] - delta_list[i]).value)

    new_config = np.zeros(adj_mat.shape)
    new_weights = {}
    for i in range(n):
        new_weights[i] = {}

    for i in range(n):
        nw = A[i, i].value
        if nw == 0:
            nw = 0.1
        new_weights[i][i] = nw
        new_config[i, i] = 1
        for j in range(i + 1, n):
            if round(PI[i, j].value) == 1:
                new_config[i, j] = round(PI[i, j].value)
                new_config[j, i] = round(PI[j, i].value)
                nw = A[i, j].value
                if nw == 0:
                    nw = 0.1
                new_weights[i][j] = nw
                new_weights[j][i] = nw
    print(new_config)
    return new_config, new_weights


def team_opt_iter(adj_mat, current_weights, covariance_matrices, omegas,
                  failed_node, ne=1):
    """
   Runs the team optimization problem

   :param adj_mat: the Adjacency matrix
   :param current_weights: current node weights
   :param covariance_data: list of each node's large covariance matrix
   :param omegas: list of each node's omega matrix
   :param ne: limit for number of edges to change
   :return: new adjacency matrix and new weights
   """
    edge_mod_limit = ne * 2
    n = adj_mat.shape[0]
    beta = 1 / n
    s = covariance_matrices[0].shape[0]
    # tol = 0.00001
    tol = 0.1
    p_size = n * s

    # Reducing Magnitude if necessary
    mag_max_val_in_matrices = magnitude(max([np.max(abs(cov))
                                             for cov in covariance_matrices]))
    if mag_max_val_in_matrices > 15:
        print('rescaling matrix magnitude, magnitude too high')
        covariance_matrices = [cov * (10 ** (-1 * mag_max_val_in_matrices))
                               for cov in covariance_matrices]

    cov_array = np.zeros((n * s, s))
    for i in range(n):
        start = i * s
        end = i * s + s
        cov_array[start:end, 0:s] = omegas[i]

    inv_cov_array = np.zeros((n * s, s))
    for i in range(n):
        start = i * s
        end = i * s + s
        inv_cov_array[start:end, 0:s] = np.linalg.inv(covariance_matrices[i])

    pos_adj_mat = []
    # Get 3 possible new adj_mats
    i = 0
    while i < 3:
        j = ne
        for x in range(i, n):
            for y in range(x+1, n):
                fnode = x + failed_node
                fnode = fnode - n if fnode >= n else fnode

                nnode = y + failed_node
                nnode = nnode - n if nnode >= n else nnode

                if adj_mat[fnode, nnode] == 1:
                    continue
                a = deepcopy(adj_mat)
                a[fnode, nnode] = 1
                a[nnode, fnode] = 1
                pos_adj_mat.append(a)

                j = j - 1
                if j == 0:
                    break
            if j == 0:
                break
        i = i + 1

    print(len(pos_adj_mat))
    best_sol_obj = None
    best_sol = None
    best_A = None
    for pi in pos_adj_mat:
        # Init Problem
        problem = pic.Problem()

        # Add Variables
        A = problem.add_variable('A', adj_mat.shape,
                                 'symmetric')  # A is an SDP var
        mu = problem.add_variable('mu', 1)  # mu is an SDP var

        Pbar = problem.add_variable('Pbar', (p_size, p_size))

        delta_list = []
        for i in range(n):
            delta_list.append(problem.add_variable('delta[{0}]'.format(i), (s, s)))

        delta_bar = problem.add_variable('delta_bar', (p_size, p_size))
        delta_array = problem.add_variable('delta_array', (p_size, s))
        # schur = problem.add_variable('schur', (p_size * 2, p_size * 2))

        # Add Params (ie constant affine expressions to help with creating constraints)
        I = pic.new_param('I', np.eye(s))
        Ibar = pic.new_param('Ibar', np.eye(p_size))
        cov_array_param = pic.new_param('covs', cov_array)
        inv_cov_array_param = pic.new_param('inv_covs', inv_cov_array)
        PI = pic.new_param('PI', pi)

        # Set Objective
        problem.set_objective('min', beta * pic.trace(Pbar))

        # Constraints

        # Setting Additional Constraint such that delta_bar elements equal elements in delta_list (with some tolerance)
        for i in range(n):
            start = i * s
            end = i * s + s
            problem.add_constraint(abs(delta_bar[start:end, start:end] - delta_list[i]) <= tol)
            if i < (n - 1):
                # Fill everything to left with 0s
                problem.add_constraint(delta_bar[start:end, end:] == np.zeros((s, (n * s) - end)))
                # Fill everything below with 0s
                problem.add_constraint(delta_bar[end:, start:end] == np.zeros(((n * s) - end, s)))

        # Setting Additional Constraint such that delta_array elements equal elements in delta_list (with some tolerance)
        for i in range(n):
            start = i * s
            end = i * s + s
            problem.add_constraint(abs(delta_array[start:end, :] - delta_list[i]) <= tol)

        # Setting Additional Constraint such that delta_bar and Pbar elements in schur variable (with some tolerance)
        # problem.add_constraint(abs(schur[0:p_size, 0:p_size] - Pbar) <= tol)
        # problem.add_constraint(abs(schur[p_size:, p_size:] - delta_bar) <= tol)
        # problem.add_constraint(schur[0:p_size, p_size:] == np.eye(p_size))
        # problem.add_constraint(schur[p_size:, 0:p_size] == np.eye(p_size))

        # Schur constraint
        # problem.add_constraint(schur >= 0)
        problem.add_constraint(((Pbar & Ibar) //
                               (Ibar & delta_bar)).hermitianized >> 0)

        # Kron constraint
        problem.add_constraint(pic.kron(A, I) * cov_array_param +
                               inv_cov_array_param == delta_array)
        # problem.add_constraint(pic.kron(A, I) * cov_array_param == delta_array)

        # Set the usual Constraints
        problem.add_constraint(mu >= 0.001)
        problem.add_constraint(mu < 1)
        problem.add_constraint(
            (A * np.ones((n, 1))) == np.ones((n, 1)))  # Constraint 1
        problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                               (1 - mu) * np.eye(n) >= A)  # Constraint 2

        for i in range(n):
            problem.add_constraint(A[i, i] > 0)  # Constraint 6
            for j in range(n):
                if i == j:
                    continue
                    # problem.add_constraint(PI[i, j] == 1.0)  # Constraint 3
                else:
                    problem.add_constraint(A[i, j] > 0)  # Constraint 7
                    problem.add_constraint(A[i, j] <= PI[i, j])  # Constraint 8

        # problem.add_constraint(
        #     abs(PI - adj_mat) ** 2 <= edge_mod_limit)  # Constraint 9

        try:
            sol = problem.solve(verbose=0, solver='mosek')
            obj = sol.value
            problem_status = problem.status
            print('status: {s}'.format(s=problem_status))
            if problem_status not in ['integer optimal', 'optimal']:
                return adj_mat, current_weights

            detI = np.linalg.det(np.dot(delta_bar.value, Pbar.value))
            if abs(detI - 1) > .1:
                print('not inverse')

            if best_sol_obj is None:
                best_sol_obj = obj
                best_sol = pi
                best_A = A
            else:
                if obj < best_sol_obj:
                    best_sol_obj = obj
                    best_sol = pi
                    best_A = A
        except SolverError as err:
            print('solver error {e}'.format(e=err))
            return pi, current_weights

    if best_A is None:
        print('no values for new weights')
        return adj_mat, current_weights

    new_config = best_sol
    new_weights = {}
    for i in range(n):
        new_weights[i] = {}

    A = best_A
    for i in range(n):
        nw = A[i, i].value
        if nw == 0:
            nw = 0.1
        new_weights[i][i] = nw
        for j in range(i + 1, n):
            if round(new_config[i, j]) == 1:
                nw = A[i, j].value
                if nw == 0:
                    nw = 0.1
                new_weights[i][j] = nw
                new_weights[j][i] = nw
    print(new_config)
    return new_config, new_weights


# Begin Branch and Bound Implementation of Team Optimization Problem

def team_opt_bnb_enum_edge_heuristic(failed_node, adj_mat):
    edge_decisions = {}
    curr_edge_decisions = {}

    neighbor_edges = []
    non_neighbor_edges = []
    other_paired_edges = []
    other_unpaired_edges = []
    for i in range(adj_mat.shape[0]):
        for j in range(i, adj_mat.shape[0]):
            if adj_mat[i, j] == 1:
                curr_edge_decisions[(i, j)] = 1
                if i == failed_node:
                    neighbor_edges.append((i, j))
                else:
                    other_paired_edges.append((i, j))
            else:
                curr_edge_decisions[(i, j)] = 0
                if i == failed_node:
                    non_neighbor_edges.append((i, j))
                else:
                    other_unpaired_edges.append((i, j))

    # Add Non Neighbor Edges
    for e in non_neighbor_edges:
        edge_decisions[e] = None

    # Add Neighbor's Non Neighbor Edges
    for e in neighbor_edges:
        for o in other_unpaired_edges:
            if e[0] in o:
                edge_decisions[o] = None

    # Add remaining unpaired edges
    for e in other_unpaired_edges:
        if e not in edge_decisions.keys():
            edge_decisions[e] = None

    # Add Neighbor edges
    for e in neighbor_edges:
        edge_decisions[e] = None

    # Add Other Paired Edges
    for e in other_paired_edges:
        edge_decisions[e] = None

    return edge_decisions, curr_edge_decisions


def team_opt_bnb_greedy_heuristic(failed_node, adj_mat, covariance_matrices, tried_nodes, use_max=True):
    det_covs = [np.linalg.det(c) for c in covariance_matrices]
    curr_node = failed_node
    det_covs_mask = deepcopy(det_covs)

    if use_max:
        for t in tried_nodes:
            det_covs_mask[t] = -np.inf
    else:
        for t in tried_nodes:
            det_covs_mask[t] = np.inf

    curr_node_saturated = False
    non_neigbor_idx = []
    while len(non_neigbor_idx) == 0:
        if curr_node_saturated:
            if use_max:
                det_covs_mask[curr_node] = -np.inf
                curr_node = det_covs_mask.index(max(det_covs_mask))
            else:
                det_covs_mask[curr_node] = np.inf
                curr_node = det_covs_mask.index(min(det_covs_mask))
            curr_node_saturated = False
        for i in range(adj_mat.shape[0]):
            if adj_mat[i, curr_node] == 0:
                non_neigbor_idx.append(i)
        if len(non_neigbor_idx) == 0:
            curr_node_saturated = True

    best_cov = None
    best_cov_id = None
    for n in non_neigbor_idx:
        if best_cov is None:
            best_cov = det_covs[n]
            best_cov_id = n
        else:
            if det_covs[n] < best_cov:
                best_cov = det_covs[n]
                best_cov_id = n

    new_adj_mat = deepcopy(adj_mat)
    new_adj_mat[curr_node, best_cov_id] = 1
    new_adj_mat[best_cov_id, curr_node] = 1

    return new_adj_mat, curr_node, best_cov_id


def team_opt_sdp(adj_mat, cov_array, inv_cov_array, s, edge_decisions, ne=1):

    n = adj_mat.shape[0]
    beta = 1 / n
    # tol = 0.00001
    tol = 0.01
    p_size = n * s
    edge_mod_limit = ne * 2

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape,
                             'symmetric')  # A is an SDP var
    mu = problem.add_variable('mu', 1)  # mu is an SDP var

    Pbar = problem.add_variable('Pbar', (p_size, p_size))

    PI = problem.add_variable('PI', adj_mat.shape, 'symmetric')

    delta_list = []
    for i in range(n):
        delta_list.append(problem.add_variable('delta[{0}]'.format(i), (s, s)))

    delta_bar = problem.add_variable('delta_bar', (p_size, p_size))
    delta_array = problem.add_variable('delta_array', (p_size, s))
    # schur = problem.add_variable('schur', (p_size * 2, p_size * 2))

    # Add Params (ie constant affine expressions to help with creating constraints)
    I = pic.new_param('I', np.eye(s))
    Ibar = pic.new_param('Ibar', np.eye(p_size))
    cov_array_param = pic.new_param('covs', cov_array)
    inv_cov_array_param = pic.new_param('inv_covs', inv_cov_array)
    # PI = pic.new_param('PI', pi)

    # Set Objective
    problem.set_objective('min', beta * pic.trace(Pbar))

    # Constraints

    # Setting Additional Constraint such that delta_bar elements equal elements in delta_list (with some tolerance)
    for i in range(n):
        start = i * s
        end = i * s + s
        problem.add_constraint(abs(delta_bar[start:end, start:end] - delta_list[i]) <= tol)
        if i < (n - 1):
            # Fill everything to left with 0s
            problem.add_constraint(delta_bar[start:end, end:] == np.zeros((s, (n * s) - end)))
            # Fill everything below with 0s
            problem.add_constraint(delta_bar[end:, start:end] == np.zeros(((n * s) - end, s)))

    # Setting Additional Constraint such that delta_array elements equal elements in delta_list (with some tolerance)
    for i in range(n):
        start = i * s
        end = i * s + s
        problem.add_constraint(abs(delta_array[start:end, :] - delta_list[i]) <= tol)

    # Setting Additional Constraint such that delta_bar and Pbar elements in schur variable (with some tolerance)
    # problem.add_constraint(abs(schur[0:p_size, 0:p_size] - Pbar) <= tol)
    # problem.add_constraint(abs(schur[p_size:, p_size:] - delta_bar) <= tol)
    # problem.add_constraint(schur[0:p_size, p_size:] == np.eye(p_size))
    # problem.add_constraint(schur[p_size:, 0:p_size] == np.eye(p_size))

    # Schur constraint
    # problem.add_constraint(schur >= 0)
    problem.add_constraint(((Pbar & Ibar) //
                           (Ibar & delta_bar)).hermitianized >> 0)

    # Kron constraint
    problem.add_constraint(pic.kron(A, I) * cov_array_param +
                           inv_cov_array_param == delta_array)
    # problem.add_constraint(pic.kron(A, I) * cov_array_param == delta_array)

    # Set the usual Constraints
    problem.add_constraint(mu >= 0.001)
    problem.add_constraint(mu < 1)
    problem.add_constraint(
        (A * np.ones((n, 1))) == np.ones((n, 1)))  # Constraint 1
    problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                           (1 - mu) * np.eye(n) >> A)  # Constraint 2

    for i in range(n):
        problem.add_constraint(A[i, i] > 0)  # Constraint 6
        for j in range(n):
            if i == j:
                problem.add_constraint(PI[i, j] == 1.0)  # Constraint 3
            else:
                problem.add_constraint(A[i, j] > 0)  # Constraint 7
                problem.add_constraint(A[i, j] <= PI[i, j])  # Constraint 8

                problem.add_constraint(PI[i, j] <= 1.0)
                problem.add_constraint(PI[i, j] >= 0.0)

    for e, d in edge_decisions.items():
        if d is not None:
            problem.add_constraint(PI[e[0], e[1]] == d)
            problem.add_constraint(PI[e[1], e[0]] == d)

    problem.add_constraint(
        abs(PI - adj_mat) ** 2 <= edge_mod_limit)  # Constraint 9

    sol = problem.solve(verbose=0, solver='mosek')
    obj = sol.value

    return obj, problem, A, PI


class BBTreeNode():
    def __init__(self, edge_decisions, curr_edge_decisions, adj_mat, current_weights, covariance_matrices, omegas, ne=1):
        n = adj_mat.shape[0]
        s = covariance_matrices[0].shape[0]

        # Reducing Magnitude if necessary
        mag_max_val_in_matrices = magnitude(max([np.max(abs(cov))
                                                 for cov in covariance_matrices]))
        if mag_max_val_in_matrices > 15:
            print('rescaling matrix magnitude, magnitude too high')
            covariance_matrices = [cov * (10 ** (-1 * mag_max_val_in_matrices))
                                   for cov in covariance_matrices]

        cov_array = np.zeros((n * s, s))
        for i in range(n):
            start = i * s
            end = i * s + s
            cov_array[start:end, 0:s] = omegas[i]

        inv_cov_array = np.zeros((n * s, s))
        for i in range(n):
            start = i * s
            end = i * s + s
            inv_cov_array[start:end, 0:s] = np.linalg.inv(covariance_matrices[i])

        self.edge_decisions = edge_decisions
        self.curr_edge_decisions = curr_edge_decisions

        self.adj_mat = adj_mat
        self.current_weights = current_weights
        self.covariance_matrices = covariance_matrices
        self.omegas = omegas
        self.ne = ne

        self.s = s
        self.cov_array = cov_array
        self.inv_cov_array = inv_cov_array

        self.children = []
        self.solved_problem = None

    def buildSolveProblem(self):
        adj_mat = self.adj_mat
        cov_array = self.cov_array
        inv_cov_array = self.inv_cov_array
        s = self.s
        obj, problem, A, PI = team_opt_sdp(adj_mat, cov_array, inv_cov_array, s, self.edge_decisions, ne=self.ne)
        self.solved_problem = A
        return obj, problem, A, PI

    def check_integrals(self, pi):
        y = (abs(pi - 1) <= 1e-2).flatten()
        z = (abs(pi - 0) <= 1e-2).flatten()
        return all([y[i] or z[i] for i in range(len(y))])

    def branch(self, next_edge):
        children = []
        for b in [0, 1]:
            edge_decisions = deepcopy(self.edge_decisions)
            edge_decisions[next_edge] = b
            n1 = BBTreeNode(edge_decisions,
                            self.curr_edge_decisions,
                            self.adj_mat,
                            self.current_weights,
                            self.covariance_matrices,
                            self.omegas,
                            ne=self.ne)
            children.append(n1)
        return children

    def bbsolve(self):
        root = self

        bestres = 1e20  # a big arbitrary initial best objective value
        bestnode = root  # initialize bestnode to the root

        res, problem, A, PI = root.buildSolveProblem()
        print("Initial Solution: ", res)
        print(PI)
        print(A)
        heap = [(res, next(counter), root)]

        nodecount = 0
        while len(heap) > 0:
            nodecount += 1  # for statistics
            print("Heap Size: ", len(heap))
            _, _, node = heappop(heap)
            obj, problem, A, PI = node.buildSolveProblem()
            print("Result: ", obj)
            print(PI)
            print(A)
            if problem.status in ['integer optimal', 'optimal']:
                if obj > bestres - 1e-3:  # even the relaxed problem sucks. forget about this branch then
                    print("Relaxed Problem Stinks. Killing this branch.")
                    pass
                elif self.check_integrals(problem.PI.value):  #if a valid solution then this is the new best
                    print("New Best Integral solution.")
                    bestres = obj
                    bestnode = node

                # otherwise, we're unsure if this branch holds promise.
                # Maybe it can't actually achieve this lower bound. So branch into it
                else:
                    changed_edges = 0
                    next_edge = None
                    for e, d in self.edge_decisions.items():
                        if e is None:
                            next_edge = e
                            break
                        if self.edge_decisions[e] != self.curr_edge_decisions[e]:
                            changed_edges += 1
                    if changed_edges >= self.ne:
                        continue

                    new_nodes = node.branch(next_edge)
                    for new_node in new_nodes:
                        # using counter to avoid possible comparisons between nodes. It tie breaks
                        heappush(heap, (res, next(counter), new_node))
        print("Nodes searched: ", nodecount)
        return bestres, bestnode, A, PI


def team_opt_bnb(adj_mat, current_weights, covariance_matrices, omegas, failed_node, ne=1):
    edge_decisions, curr_edge_decisions = team_opt_bnb_enum_edge_heuristic(failed_node, adj_mat)
    root = BBTreeNode(edge_decisions, curr_edge_decisions, adj_mat, current_weights, covariance_matrices, omegas, ne=ne)
    bestres, bestnode, A, PI = root.bbsolve()
    print("best solution value: ", bestres)

    if A is None:
        return adj_mat, current_weights

    new_config = PI

    n = adj_mat.shape[0]
    new_weights = {}
    for i in range(n):
        new_weights[i] = {}

    for i in range(n):
        nw = A[i, i].value
        if nw == 0:
            nw = 0.1
        new_weights[i][i] = nw
        for j in range(i + 1, n):
            if round(new_config[i, j]) == 1:
                nw = A[i, j].value
                if nw == 0:
                    nw = 0.1
                new_weights[i][j] = nw
                new_weights[j][i] = nw
    print(new_config)
    return new_config, new_weights

