import csv
import cvxopt as cvx
import math
import numpy as np
import os
import picos as pic
import platform
from scipy.linalg import block_diag


def magnitude(x):
    return int(math.log10(x))


def agent_opt(adj_mat, current_weights, covariance_data, ne=1, failed_node=None):
    """
    Runs the agent optimization problem

    :param adj_mat: the Adjacency matrix
    :param current_weights: current node weights
    :param covariance_data: list of each node's large covariance matrix
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
    magnitude_covs = [magnitude(cov) for cov in covariance_data]
    if max(magnitude_covs) > 17:
        covariance_data = [cov * 1e-17 for cov in covariance_data]

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
    if problem_status != 'integer optimal':
        return adj_mat, current_weights

    new_config = np.zeros(adj_mat.shape)
    new_weights = {}
    for i in range(n):
        new_weights[i] = {}

    for i in range(n):
        new_weights[i][i] = A[i, i].value
        new_config[i, i] = 1
        for j in range(i + 1, n):
            new_config[i, j] = round(PI[i, j].value)
            new_config[j, i] = round(PI[j, i].value)
            new_weights[i][j] = A[i, j].value
            new_weights[j][i] = A[j, i].value
    return new_config, new_weights


def team_opt(adj_mat, current_weights, covariance_matrices, ne=1):
    A = adj_mat
    n = adj_mat.shape[0]
    for c, cov in enumerate(covariance_matrices):
        f_name = 'misdp_data/inverse_covariance_matrices/{c}.csv'
        np.savetxt(f_name.format(c=c), np.asarray(cov), delimiter=",")

    np.savetxt('misdp_data/adj_mat.csv', A, delimiter=",")
    if platform.system() == 'Linux':
        mat_command = 'matlab '
    else:
        mat_command = '/Applications/MATLAB_R2019a.app/bin/matlab '
    matlab_string = mat_command + '-nodesktop -nosplash -r "MISDP_new_copy({e});exit;"'.format(
        e=ne)
    print('Starting Up MATLAB')
    os.system(matlab_string)
    print('Reading MATLAB results')

    # if os.path.exists('misdp_data/fail.txt'):
    #     os.system(matlab_string)
    #     os.unlink('misdp_data/fail.txt')

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

    return new_A, new_weights


def team_opt2(adj_mat, current_weights, covariance_matrices, ne=1):
    """
   Runs the team optimization problem

   :param adj_mat: the Adjacency matrix
   :param current_weights: current node weights
   :param covariance_data: list of each node's large covariance matrix
   :param ne: limit for number of edges to change
   :param failed_node: the node that fails
   :return: new adjacency matrix and new weights
   """
    edge_mod_limit = ne * 2
    n = adj_mat.shape[0]
    beta = 1 / n

    # Reducing Magnitude if necessary
    mag_max_val_in_matrices = magnitude(max([np.max(cov)
                                         for cov in covariance_matrices]))
    if mag_max_val_in_matrices > 17:
        covariance_data = [cov * 1e-17 for cov in covariance_matrices]

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape,
                             'symmetric')  # A is an SDP var
    PI = problem.add_variable('PI', adj_mat.shape,
                              'binary')  # PI is a binary var
    mu = problem.add_variable('mu', 1)  # mu is an SDP var

    Pbar = problem.add_variable('Pbar', (n * covariance_matrices[0].shape[0],
                                         n * covariance_matrices[0].shape[1]),
                                'complex')

    delta_bar_list = []
    for i in range(n):
        delta_bar_list.append(
            problem.add_variable('delta[{0}]'.format(i),
                                 covariance_matrices[0].shape,
                                 'complex'))
    print(delta_bar_list)

    # TODO: check if this is going to work
    delta_bar = problem.add_variable('delta_bar',
                                     (n * covariance_matrices[0].shape[0],
                                      n * covariance_matrices[0].shape[1]),
                                     'complex')
    """
    Setting Additional Constraint such that delta_bar elements equal 
    elements in delta_bar_list 
    """

    # delta_bar = []
    # for i in range(n):
    #     for j in range(n):
    #         if i == j:
    #             delta_bar.append(delta_bar_list[i])
    #         else:
    #             delta_bar.append(np.zeros(covariance_matrices[0].shape))
    # print(delta_bar)


    # # Set Objective
    # problem.set_objective('min',
    #                       -1 * node_bin * A * np.array(covariance_data).T)
    #
    # # Set Constraints
    # problem.add_constraint(mu >= 0.001)
    # problem.add_constraint(mu < 1)
    # problem.add_constraint(
    #     (A * np.ones((n, 1))) == np.ones((n, 1)))  # Constraint 1
    # problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
    #                        (1 - mu) * np.eye(n) >= A)  # Constraint 2
    #
    # for i in range(n):
    #     problem.add_constraint(A[i, i] > 0)  # Constraint 6
    #     for j in range(n):
    #         if i == j:
    #             problem.add_constraint(PI[i, j] == 1.0)  # Constraint 3
    #         else:
    #             problem.add_constraint(A[i, j] > 0)  # Constraint 7
    #             problem.add_constraint(A[i, j] <= PI[i, j])  # Constraint 8
    #
    # problem.add_constraint(
    #     abs(PI - adj_mat) ** 2 <= edge_mod_limit)  # Constraint 9
    #
    # problem.solve(verbose=0, solver='mosek')
    # problem_status = problem.status
    # print('status: {s}'.format(s=problem_status))
    # if problem_status != 'integer optimal':
    #     return adj_mat, current_weights
    #
    # new_config = np.zeros(adj_mat.shape)
    # new_weights = {}
    # for i in range(n):
    #     new_weights[i] = {}
    #
    # for i in range(n):
    #     new_weights[i][i] = A[i, i].value
    #     new_config[i, i] = 1
    #     for j in range(i + 1, n):
    #         new_config[i, j] = round(PI[i, j].value)
    #         new_config[j, i] = round(PI[j, i].value)
    #         new_weights[i][j] = A[i, j].value
    #         new_weights[j][i] = A[j, i].value
    # return new_config, new_weights


# adj_mat = np.array([[1, 1, 0],
#                     [1, 1, 1],
#                     [0, 1, 1]])
#
# current_weights = {0: {0: .67, 1: .33, 2: 0.},
#                    1: {0: .33, 1: .33, 2: 0.33},
#                    2: {0: 0., 1: .33, 2: 0.67}}
#
# c0 = np.diag((0.01, 0.01, 0.01, 0.01))
# c1 = np.diag((0.01, 0.01, 0.01, 0.01))
# c2 = np.diag((0.01, 0.01, 0.01, 0.01))
# covariance_matrices = [c0, c1, c2]
#
# print(adj_mat)
# print(current_weights)
# print(covariance_matrices)
#
# team_opt2(adj_mat, current_weights, covariance_matrices)
