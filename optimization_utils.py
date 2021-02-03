import math
import numpy as np
import picos as pic


def magnitude(x):
    return int(math.log10(abs(x)))


def agent_opt(adj_mat, current_weights, covariance_data, ne=1,
              failed_node=None, edge_decisions=None):
    """
    Runs the agent optimization problem
    :param adj_mat: the Adjacency matrix
    :param current_weights: current node weights
    :param covariance_data: list of each node's large covariance matrix
    :param ne: limit for number of edges to change
    :param failed_node: the node that fails
    :param edge_decisions: dictionary, if provided, the set of edges to set
        as 1 or 0 in their corresponding entries in PI
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
        covariance_data = [cov * (10 ** (-1 * max(magnitude_covs)))
                           for cov in covariance_data]

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape, 'symmetric')
    PI = problem.add_variable('PI', adj_mat.shape, 'symmetric')
    mu = problem.add_variable('mu', 1)

    # Set Objective
    problem.set_objective('min',
                          -1 * node_bin * A * np.array(covariance_data).T)

    # Set Constraints
    problem.add_constraint(mu >= 0.001)
    problem.add_constraint(mu < 1)
    problem.add_constraint(
        (A * np.ones((n, 1))) == np.ones((n, 1)))
    problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                           (1 - mu) * np.eye(n) >= A)

    for i in range(n):
        problem.add_constraint(A[i, i] > 0)
        for j in range(n):
            if i == j:
                problem.add_constraint(PI[i, j] == 1.0)
            else:
                problem.add_constraint(PI[i, j] <= 1.0)
                problem.add_constraint(PI[i, j] >= 0.0)

                problem.add_constraint(A[i, j] > 0)
                problem.add_constraint(A[i, j] <= PI[i, j])

    if edge_decisions is not None:
        # Ensures the set edge_decisions are maintained in PI
        for e, d in edge_decisions.items():
            if d is not None:
                problem.add_constraint(PI[e[0], e[1]] == d)
                problem.add_constraint(PI[e[1], e[0]] == d)

        # Ensures the previous edges are maintained in PI
        for i in range(n):
            for j in range(n):
                if adj_mat[i, j] == 1:
                    problem.add_constraint(PI[i, j] == 1.0)

    # problem.add_constraint(
    #     abs(PI - adj_mat) ** 2 <= edge_mod_limit)

    try:
        problem.solve(verbose=0, solver='mosek')
        # problem_status = problem.status
        # print(problem_status)

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

        return problem, problem.obj_value(), new_config, new_weights
    except Exception as e:
        print('solve error')
        print(e)
        return problem, 'infeasible', adj_mat, current_weights


def team_opt(adj_mat, current_weights, covariance_matrices, how='geom', ne=1,
             edge_decisions=None):
    """
   Runs the team optimization problem
   :param adj_mat: the Adjacency matrix
   :param current_weights: current node weights
   :param covariance_matrices: list of each node's large covariance matrix
   :param how: string that denotes fusion method
   :param ne: limit for number of edges to change
   :param edge_decisions: dictionary, if provided, the set of edges to set
        as 1 or 0 in their corresponding entries in PI
   :return: new adjacency matrix and new weights
   """
    n = adj_mat.shape[0]
    beta = 1 / n
    tol = 0.1
    s = covariance_matrices[0].shape[0]
    p_size = n * s
    edge_mod_limit = ne * 2

    # Init Problem
    problem = pic.Problem()

    # Add Variables
    A = problem.add_variable('A', adj_mat.shape, 'symmetric')
    mu = problem.add_variable('mu', 1)
    Pbar = problem.add_variable('Pbar', (p_size, p_size))
    PI = problem.add_variable('PI', adj_mat.shape, 'symmetric')

    delta_list = []
    for i in range(n):
        delta_list.append(problem.add_variable('delta[{0}]'.format(i), (s, s)))

    delta_bar = problem.add_variable('delta_bar', (p_size, p_size))
    delta_array = problem.add_variable('delta_array', (p_size, s))

    # Add Params (ie constant affine expressions to help with creating constraints)
    cov_array = np.zeros((n * s, s))
    for i in range(n):
        start = i * s
        end = i * s + s
        cov_array[start:end, 0:s] = covariance_matrices[i]

    I = pic.new_param('I', np.eye(s))
    Ibar = pic.new_param('Ibar', np.eye(p_size))
    cov_array_param = pic.new_param('covs', cov_array)


    # Set Objective
    if how == 'geom':
        problem.set_objective('min', pic.trace(Pbar))
    else:
        problem.set_objective('min', pic.trace(delta_bar))

    # Constraints

    # Setting Additional Constraint such that delta_bar elements equal elements in delta_list (with some tolerance)
    for i in range(n):
        start = i * s
        end = i * s + s
        problem.add_constraint(abs(
            delta_bar[start:end, start:end] - delta_list[i]) <= tol)
        if i < (n - 1):
            # Fill everything to left with 0s
            problem.add_constraint(
                delta_bar[start:end, end:] == np.zeros(
                    (s, (n * s) - end)))
            # Fill everything below with 0s
            problem.add_constraint(
                delta_bar[end:, start:end] == np.zeros(
                    ((n * s) - end, s)))

    # Setting Additional Constraint such that delta_array elements equal elements in delta_list (with some tolerance)
    for i in range(n):
        start = i * s
        end = i * s + s
        problem.add_constraint(
            abs(delta_array[start:end, :] - delta_list[i]) <= tol)

    if how == 'geom':
        # Schur constraint
        problem.add_constraint(((Pbar & Ibar) //
                                (Ibar & delta_bar)).hermitianized >> 0)

    # Kron constraint
    problem.add_constraint(pic.kron(A, I) * cov_array_param == delta_array)

    problem.add_constraint(mu >= 0.001)
    problem.add_constraint(mu < 1)
    problem.add_constraint(
        (A * np.ones((n, 1))) == np.ones((n, 1)))
    problem.add_constraint((beta * np.dot(np.ones(n).T, np.ones(n))) +
                           (1 - mu) * np.eye(n) >= A)

    for i in range(n):
        problem.add_constraint(A[i, i] > 0)
        for j in range(n):
            if i == j:
                problem.add_constraint(PI[i, j] == 1.0)
            else:
                problem.add_constraint(PI[i, j] <= 1.0)
                problem.add_constraint(PI[i, j] >= 0.0)

                problem.add_constraint(A[i, j] > 0)
                problem.add_constraint(A[i, j] <= PI[i, j])

    if edge_decisions is not None:
        # Ensures the set edge_decisions are maintained in PI
        for e, d in edge_decisions.items():
            if d is not None:
                problem.add_constraint(PI[e[0], e[1]] == d)
                problem.add_constraint(PI[e[1], e[0]] == d)

        # Ensures the previous edges are maintained in PI
        for i in range(n):
            for j in range(n):
                if adj_mat[i, j] == 1:
                    problem.add_constraint(PI[i, j] == 1.0)

    # problem.add_constraint(
    #     abs(PI - adj_mat) ** 2 <= edge_mod_limit)

    try:
        problem.solve(verbose=0, solver='mosek')
        # problem_status = problem.status
        # print(problem_status)

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

        return problem, problem.obj_value(), new_config, new_weights
    except Exception as e:
        print('solve error')
        print(e)
        return problem, 'infeasible', adj_mat, current_weights
