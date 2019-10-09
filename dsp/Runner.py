import copy
import time
from scipy.spatial.distance import cdist, pdist

import numpy as np

from dsp.Problem import Problem
from dsp.Solver import DPSolver
import queue
from collections import deque

ALPHA = np.arange(1, 64)
np.random.seed(10)


def sequence_search(root, available, problem, func=None, height=0, size=10000):
    """

    :param num: ID of the ship
    :param available: Set of available ship ids
    :param visited: set of visited ship ids
    :return:
    """

    # Set the harbor
    root.next(0)
    h = 0  # Height counter
    Q = deque([root, None])
    infeasible = 0
    total = 0

    best_dist = [0]
    level_best_dist = 1e10

    best_solver = [root]
    level_best_solver = None
    q_size = [1]
    while len(Q) > 0:

        current = Q.popleft()

        if current is None:
            sorted_q = sorted(Q, key=lambda x: x.states[-1].distances[0])
            if len(sorted_q) > size:
                sorted_q = sorted_q[:size]

                Q = deque(sorted_q)
            # Next level is starting
            Q.append(None)

            print(f"Finished Processing Level # {h}")
            h += 1
            q_size.append(len(Q))
            best_dist.append(level_best_dist)
            level_best_dist = 1e10
            best_solver.append(level_best_solver)
            # with open(f"/home/yash/PycharmProjects/DSP/results/res1.F", 'w+') as fp:
            #     fp.write(f"Exploration: {np.inf}, Truncation: {10000}\n")
            #     fp.write(f"Level # {h} \n")
            #     fp.write("\n".join(f"Level {i}: {str(item)}, Dist: {best_dist[i]}" for i, item in enumerate(best_solver)))
            if h >= height:
                break
            continue

        avail = available - set(current.seq)
        if func:
            if current is not root:
                avail = func(avail=avail, current=current, problem=problem)


        for s in avail:
            total += 1
            sol = copy.copy(current)
            if len(sol.states) > 1:
                sol.pop_state()

            if sol.next(s) is False: continue
            if sol.next(0) is False: continue

            last_state = sol.get_last_state()
            if not last_state or len(last_state) == 0:
                infeasible += 1
                continue

            sol.feasible = True  # Set the DP solver to feasible

            path_distance = last_state.distances[0]
            if path_distance < level_best_dist:
                level_best_dist = path_distance
                level_best_solver = sol
            if h <= height:
                Q.append(sol)
    print(f"Infeasible Count: {infeasible}")
    return best_dist, best_solver, q_size, total


def distance_based(avail, current, problem):

        # We use some metric to limit the expansion of a node to 10

        t, pos, _id = current.get_most_recent_time_pos()
        current_ship = problem.get_ship(_id)
        pos = current_ship.get_positions(range=(t, current_ship.get_times()[1]))

        s_id = []
        abs_dist = []

        for p in avail:
            ship = problem.get_ship(p)

            ship_pos = ship.get_positions(range=(t+1, ship.get_times()[1]))

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


if __name__ == "__main__":
    # Time frame
    T = 6

    # Data
    x_data = np.genfromtxt("/home/yash/PycharmProjects/DSP/data/x.csv", delimiter=",")
    y_data = np.genfromtxt("/home/yash/PycharmProjects/DSP/data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data, T=6)
    ALPHA = set(P.in_working_area)
    root = DPSolver(P, seq=[])
    start = time.time()
    best_dist, best_solv, q_size, total = sequence_search(root=root, available=set(ALPHA), problem=P, height=5)  # Exhaustive
    # best_dist, best_solv, q_size, total = sequence_search(root=root, available=set(ALPHA), problem=P, func=distance_based, height=23)
    end = time.time()
    print(f"{end-start}")
    print(total)
