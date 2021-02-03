from copy import deepcopy
from heapq import *
import itertools
import numpy as np
from optimization_utils import agent_opt, team_opt

counter = itertools.count()


class BBTreeNode():
    def __init__(self, possible_edge_decisions, current_edge_decisions,
                 adj_mat, weights, covariance_data,
                 edge_mode_limit=1, failed_node=None, opt='agent'):
        """
        Initialize class for solving the MISDP problems defined in project using
        Branch and Bound algorithm.
        Much of this code was inspired by the Branch and Bound solver implemented
        in this blog post:
            https://www.philipzucker.com/a-basic-branch-and-bound-solver-in-python-using-cvxpy/
        :param possible_edge_decisions: dictionary, the set of possible edges
            in the network to try to add to minimize the objective
        :param current_edge_decisions: dictionary, the set of existing edges
            in the network
        :param adj_mat: ndarray, the closed adjacency matrix, A
        :param weights: current weights of the network
        :param covariance_data: ndarray, the covariance data from the network
        :param edge_mod_limit: int, the maximal number of edges to be modified
            by the problem, default 1
        """
        self.possible_edge_decisions = possible_edge_decisions
        self.current_edge_decisions = current_edge_decisions
        self.adj_mat = adj_mat
        self.weights = weights
        self.cov_data = covariance_data
        self.edge_mode_limit = edge_mode_limit
        self.failed_node = failed_node
        self.opt = opt

        self.n = adj_mat.shape[0]

        self.children = []
        self.solved_problem = None

    def buildSolveProblem(self, fuse_method=None):
        """
        Builds and solves the MISDP problems defined in the project using the
        agent_opt or team_opt function.
        :param fuse_method: string, the arithmetic or geometric fusion method
        :return: the results of optimization problem
        """

        if self.opt == 'agent':
            problem, obj, PI, weights = \
                agent_opt(self.adj_mat,
                          self.weights,
                          self.cov_data,
                          ne=self.edge_mode_limit,
                          failed_node=self.failed_node,
                          edge_decisions=self.possible_edge_decisions)

        else:
            problem, obj, PI, weights = \
                team_opt(self.adj_mat,
                         self.weights,
                         self.cov_data,
                         how=fuse_method,
                         ne=self.edge_mode_limit,
                         edge_decisions=self.possible_edge_decisions)

        self.solved_problem = problem
        return problem, obj, PI, weights

    def check_integrals(self, PI):
        """
        Checks if PI returns close to an integral solution
        :param PI: ndarray, the adjacency matrix returned by ConfigGen
        :return: boolean
        """
        n = self.adj_mat.shape[0]
        for i in range(n):
            for j in range(n):
                if i == j:
                    if PI[i, j] != 1.0:
                        return True
                else:
                    x = PI[i, j]
                    y = (abs(x - 1) <= 1e-2)
                    z = (abs(x - 0) <= 1e-2)
                    if not (y or z):
                        return False
        return True

    def branch(self, next_edge):
        """
        Creates the branch to search based on the given next_edge to test
        :param next_edge: tuple, the pair of robots for which to create a
            connecting edge
        :return: new BBTreeNode object
        """
        children = []
        for b in [0, 1]:
            edge_decisions = deepcopy(self.possible_edge_decisions)
            edge_decisions[next_edge] = b
            n1 = BBTreeNode(edge_decisions,
                            self.current_edge_decisions,
                            self.adj_mat,
                            self.res_mat,
                            edge_mode_limit=self.edge_mode_limit)
            children.append(n1)
        return children

    def bbsolve(self, fuse_method='geom'):
        """
        Executes the branch and bound algorithm
        :param fuse_method: string, the arithmetic or geometric fusion method
        :return: four values
            1) the best objective value found
            2) the node that returns the best objective value
            3) the adjacency matrix PI found from solving the problem returned
                from the best node
            4) IF self.positions = False, the decision variable for the Laplacian, L, as ndarray
            ELSE, the decision variable for the inter-robot distances, D, as ndarray
        """
        root = self

        bestobj = 1e20  # a big arbitrary initial best objective value
        bestnode = root  # initialize bestnode to the root

        problem, obj, PI, weights = root.buildSolveProblem(fuse_method=fuse_method)

        # If root problem is infeasible, do not go further,
        # return the current configuration
        if obj == 'infeasible':
            return obj, bestnode, PI, weights

        # Add root node to the search heap
        heap = [(obj, next(counter), root)]

        bestPI = PI
        best_weights = weights

        nodecount = 0
        while len(heap) > 0:
            nodecount += 1
            _, _, node = heappop(heap)
            problem, obj, PI, weights = root.buildSolveProblem()
            problem_status = problem.status
            if nodecount == 1:
                bestobj = obj

            # check if L is indeed positive semi-definite
            beta = 1/self.n
            eig_vals = np.linalg.eigvals(beta * np.ones((self.n, self.n)) + PI)
            sdp_check = np.all(eig_vals > 0)

            if problem_status == 'solver error':
                continue
            if problem_status in ['optimal']:
                if (obj > bestobj - 1e-3) and (nodecount > 1):
                    # if returned objective from this node is not better and
                    # there are more nodes to search,
                    # do not continue searching this branch
                    continue
                elif not sdp_check:
                    # if graph represented by resulting PI matrix not connected,
                    # do not continue searching this branch
                    continue
                elif self.check_integrals(PI):
                    # if node returns a valid solution,
                    # set this as the new best solution
                    bestobj = obj
                    bestnode = node
                    bestPI = PI
                    best_weights = weights

                # Otherwise branch off of this node and continue searching
                else:
                    changed_edges = 0
                    next_edge = None
                    for e, d in node.possible_edge_decisions.items():
                        if d is None:
                            # If this edge decision has not been explored,
                            # make it one of the next edges to explore
                            next_edge = e
                            break
                        if node.possible_edge_decisions[e] != self.current_edge_decisions[e]:
                            # If this edge decision has already been made
                            # note it as one of the changed edges
                            changed_edges += 1
                        elif node.possible_edge_decisions[e] == self.current_edge_decisions[e]:
                            continue
                    if changed_edges >= self.edge_mode_limit:
                        # if already exceeded the number of edges to change
                        # in this iteration, do not proceed with branching
                        continue

                    if next_edge is None:
                        # if no new edges to explore,
                        # do not proceed with branching
                        continue

                    # Create the new branches
                    new_nodes = node.branch(next_edge)
                    for new_node in new_nodes:
                        # add branches to search heap based on the objective
                        # value found by its parent
                        heappush(heap, (obj, next(counter), new_node))

        return bestobj, bestnode, bestPI, best_weights


def get_possible_edges(failed_node, adj_mat):
    """
    Enumerates all possible edges to try to add to the network starting
    with the robot that failed.
    :param failed_node: int, the node representing the robot that
        experienced failure
    :param adj_mat: ndarray, the current adjacency matrix
    :return: two values
        1) dictionary of possible edges to add
        2) existing edges in the network
    """
    possible_edge_decisions = {}
    current_edge_decisions = {}

    neighbor_edges = []
    non_neighbor_edges = []
    other_paired_edges = []
    other_unpaired_edges = []
    for i in range(adj_mat.shape[0]):
        for j in range(i+1, adj_mat.shape[0]):
            if adj_mat[i, j] == 1:
                current_edge_decisions[(i, j)] = 1
                if i == failed_node:
                    neighbor_edges.append((i, j))
                else:
                    other_paired_edges.append((i, j))
            else:
                current_edge_decisions[(i, j)] = 0
                if i == failed_node:
                    non_neighbor_edges.append((i, j))
                else:
                    other_unpaired_edges.append((i, j))

    # Add Non Neighbor Edges to Possible Edge Decisions
    for e in non_neighbor_edges:
        possible_edge_decisions[e] = None

    # Add Neighbor's Non Neighbor Edges
    for e in neighbor_edges:
        for o in other_unpaired_edges:
            if e[0] in o:
                possible_edge_decisions[o] = None

    # Add remaining unpaired edges
    for e in other_unpaired_edges:
        if e not in possible_edge_decisions.keys():
            possible_edge_decisions[e] = None

    edge_keys = list(possible_edge_decisions.keys())
    np.random.shuffle(edge_keys)
    possible_edge_decisions_shuffle = {e: possible_edge_decisions[e]
                                       for e in edge_keys}

    return possible_edge_decisions_shuffle, current_edge_decisions

