from copy import deepcopy
import numpy as np
import quadpy
import platform


def generate_coords(new_config, current_coords, fov,
                    bbox=np.array([(-50, 50), (-50, 50), (10, 100)]),
                    delta=10, safe_dist=10, connect_dist=25, k=-0.1, steps=1000,
                    lax=True):
    """
    Uses Simulated Annealing to generate new coordinates given new config

    :param new_config: new adjacency matrix
    :param current_coords: dictionary of nodes and positions
    :param fov: Dictionary of Field of View of nodes
    :param target_estimate: centroid of PHD

    :param bbox: region bounding box
    :param delta: delta is max movement
    :param safe_dist: safe distances between nodes
    :param connect_dist: connect distances between nodes
    :param steps: simulated annealing steps
    :return:
    """

    invalid_iters_limit = 3
    # if platform.system() == 'Linux':
    #     invalid_iters_limit = 10
    #     steps = 10000
    # else:
    #     invalid_iters_limit = 5

    # Simulated Annealing
    H = np.logspace(1, 3, steps)
    temperature = np.logspace(1, -8, steps)

    new_coords = current_coords
    valid_config = False
    invalid_configs = 0
    while not valid_config:
        for i in range(steps):
            T = temperature[i]
            propose_coords = propose(new_coords, delta)
            current_E = energyCoverage(new_config, new_coords, fov,
                                       H[i], k, safe_dist, connect_dist, bbox)
            propose_E = energyCoverage(new_config, propose_coords, fov,
                                       H[i], k, safe_dist, connect_dist, bbox)
            if propose_E < current_E:
                new_coords = deepcopy(propose_coords)
            else:
                p_accept = np.exp((-1 * (propose_E - current_E)) / T)
                accept_criteria = np.random.uniform(0, 1)
                if accept_criteria < p_accept:
                    new_coords = deepcopy(propose_coords)
            del propose_coords

        valid_config = isValidConfig(new_config, new_coords,
                                     safe_dist, connect_dist, bbox)
        if not valid_config:
            invalid_configs = invalid_configs + 1
        if invalid_configs > invalid_iters_limit:
            print('could not find valid config')
            if lax:
                return new_coords
            else:
                return False

    return new_coords


def propose(current_coords, delta):
    """
    Propose New Coordinates
    :param current_coords: dictionary of nodes and positions
    :param delta: delta is max movement
    :return: dictionary of proposed coordinates for nodes
    """
    propose_coords = deepcopy(current_coords)
    node = np.random.choice(list(current_coords.keys()))
    dir = np.random.choice(3)
    d = np.random.uniform(-1 * delta, delta)

    old_pos = current_coords[node][dir]
    propose_coords[node][dir] = old_pos + d

    return propose_coords

# TODO: pass R values to be translated to sigma values (inversely proportional) in Ho and Hc funcs
# TODO: return overall coverage quality
def energyCoverage(config, propose_coords, fov,
                   H, k, safe_dist, connect_dist, bbox):
    """
    Get Energy, try to congregate around centroid of PHD

    :param config: New Configuration
    :param propose_coords: Proposed Coordinates
    :param fov: Dictionary of Field of View (ie. radius of FoV) of nodes
    :param H: entropy factor for simulated annealing
    :param k: some scaling factor for simulated annealing
    :param safe_dist: safe distances between nodes
    :param connect_dist: connect distances between nodes
    :param bbox: region bounding box
    :return: Energy value
    """
    total_coverage = 0
    coverage_penalties = 0
    n = len(fov)
    for i in range(n):
        total_coverage += Hc(propose_coords[i], fov[i])
        for j in range(i, n):
            coverage_penalties += Ho(propose_coords[i], propose_coords[j],
                                     fov[i], fov[j])

    sum_box = 0
    sum_safe = 0
    sum_conn = 0
    bbox = bbox

    n = len(propose_coords)
    for i in range(n):
        pos = propose_coords[i]
        sum_box_node = 0

        for d in range(len(bbox)):
            sum_box_node = sum_box_node + (ph(pos[d] - bbox[d, 1], H) +
                                           ph(bbox[d, 0] - pos[d], H))
        sum_box = sum_box + sum_box_node

        for j in range(i + 1, n):
            d = np.linalg.norm(propose_coords[i] - propose_coords[j])

            if config[i, j] > 0:
                sum_safe = sum_safe + (ph(safe_dist - d, H) +
                                       ph(d - connect_dist, H))
            else:
                sum_conn = sum_conn + (ph(connect_dist - d, H))

    return (k * total_coverage) + coverage_penalties + \
           sum_box + sum_safe + sum_conn


def isValidConfig(config, coords, safe_dist, connect_dist, bbox):
    """
    Checks if proposed config is valid
    :param config: proposed config
    :param coords: dictionary or coordinates of nodes
    :param safe_dist: safe distances between nodes
    :param connect_dist: connect distances between nodes
    :param bbox: region bounding box
    :return:
    """
    n = len(coords)

    for i in range(n):
        # Check Position is within BBox
        x, y, z = coords[i]
        if not (bbox[0, 0] <= x <= bbox[0, 1]):
            print("x pos not in bbox")
            return False
        if not (bbox[1, 0] <= y <= bbox[1, 1]):
            print("y pos not in bbox")
            return False
        if not (bbox[2, 0] <= z <= bbox[2, 1]):
            print("z pos not in bbox")
            return False

        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])

            # If Neighbors, check if between safe_dist and connect_dist
            if config[i, j] > 0:
                if not (safe_dist <= d <= connect_dist):
                    print(
                        'too close or too far from connection, {i}-{j}'.format(
                            i=i, j=j))
                    return False
            # If not, check if greater than connect_dist
            else:
                if not (connect_dist <= d):
                    print(
                        'too close to neighbor, {i}-{j}'.format(i=i, j=j))
                    return False
    return True


def ph(x, H):
    """
    Temperature func for Simulated Annealing
    """
    if x < 0:
        return 0
    else:
        return np.exp(H * x)


def Hc(drone_pos, fov_radius, focal_length=0.04, sigma=0.3, R=3, kappa=4):
    x = drone_pos[0]
    y = drone_pos[1]
    z = drone_pos[2]
    r = fov_radius

    a1 = (focal_length / np.sqrt(
        (focal_length ** 2) + (fov_radius ** 2))) ** kappa
    b1 = z
    c1 = focal_length / np.sqrt((focal_length ** 2) + (fov_radius ** 2))

    a2 = (focal_length / np.sqrt((focal_length ** 2) +
                                 (fov_radius ** 2))) ** kappa
    b2 = R
    c2 = 2 * (sigma ** 2)

    scheme = quadpy.disk.lether(6)
    val = scheme.integrate(lambda p: (fpers(p[0], a1, b1, c1) *
                                      fres(p[0], a2, b2, c2)),
                           [x, y], r
                           )
    return val


def Ho(drone1_pos, drone2_pos, fov1_radius, fov2_radius, focal_length=0.04, sigma=0.3, R=3, kappa=4):
    mid_x = (drone1_pos[0] + drone2_pos[0]) / 2.
    mid_y = (drone1_pos[1] + drone2_pos[1]) / 2.

    drone1_pos_0_height = deepcopy(drone1_pos)
    drone1_pos_0_height[2] = 0
    drone2_pos_0_height = deepcopy(drone2_pos)
    drone2_pos_0_height[2] = 0
    d = np.linalg.norm(drone1_pos_0_height - drone2_pos_0_height)

    overlap = 2 * max(fov1_radius, fov2_radius) - d

    # Drone 1 vars
    a1 = (focal_length / np.sqrt(
        (focal_length ** 2) + (fov1_radius ** 2))) ** kappa
    b1 = drone1_pos[2]
    c1 = focal_length / np.sqrt((focal_length ** 2) + (fov1_radius ** 2))

    a2 = (focal_length / np.sqrt((focal_length ** 2) +
                                 (fov1_radius ** 2))) ** kappa
    b2 = R
    c2 = 2 * (sigma ** 2)

    # Drone 2 vars
    a3 = (focal_length / np.sqrt(
        (focal_length ** 2) + (fov2_radius ** 2))) ** kappa
    b3 = drone2_pos[2]
    c3 = focal_length / np.sqrt((focal_length ** 2) + (fov2_radius ** 2))

    a4 = (focal_length / np.sqrt((focal_length ** 2) +
                                 (fov2_radius ** 2))) ** kappa
    b4 = R
    c4 = 2 * (sigma ** 2)

    scheme = quadpy.disk.lether(6)
    val1 = scheme.integrate(lambda x: (fpers(x[0], a1, b1, c1) *
                                       fres(x[0], a2, b2, c2)),
                           [mid_x, mid_y], overlap
                           )

    val2 = scheme.integrate(lambda x: (fpers(x[0], a3, b3, c3) *
                                       fres(x[0], a4, b4, c4)),
                            [mid_x, mid_y], overlap
                            )
    return min(val1, val2)


def fpers(p, a, b, c):
    if p.all() == 0:
        return a * b
    return a * ((b / p) - c)


def fres(p, a, b, c):
    return a * np.exp(-1 * ((p - b) ** 2) / c)


