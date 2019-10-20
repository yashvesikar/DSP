from scipy.spatial.distance import cdist
import numpy as np

def distance_exploration(avail, current, problem):
    """
    Distance based exploration
    :param avail:
    :param current:
    :param problem:
    :return:
    """
    # We use some metric to limit the expansion of a node to 10

    t, pos, _id = current.get_most_recent_time_pos()
    current_ship = problem.get_ship(_id)
    pos = current_ship.get_positions(range=(t, current_ship.get_times()[1]))

    s_id = []
    abs_dist = []

    for p in avail:
        ship = problem.get_ship(p)

        ship_pos = ship.get_positions(range=(t + 1, ship.get_times()[1]))

        if len(ship_pos):
            dist = cdist(pos, ship_pos)
            s_id.append(p)
            abs_dist.append(np.min(dist))

    s_id = np.array(s_id)
    if len(s_id) > 15:
        next_lvl = s_id[np.argsort(abs_dist)][:10]
    else:
        next_lvl = s_id

    avail = set(next_lvl)
    return avail


def exhaustive_exploration(avail, current, problem):
    return avail


def select_exploration(method='exhaustive'):

    methods = {"distance": distance_exploration,
               "exhaustive": exhaustive_exploration}

    if methods.get(method):
        return methods.get(method)
