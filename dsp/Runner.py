import copy

import numpy as np

from dsp.Problem import Problem
from dsp.Solver import DPSolver
from dsp.Visualize import Visualizer
import queue

ALPHA = np.arange(1, 64)


def exhaustive(root, available, problem, height=0):
    """

    :param num: ID of the ship
    :param available: Set of available ship ids
    :param visited: set of visited ship ids
    :return:
    """

    # Set the harbor
    root.next(0)
    h = 0  # Height counter
    Q = queue.Queue()
    Q.put(root)
    Q.put(None)
    infeasible = 0
    total = 0
    level_best = [1e10]
    level_best_solvers = [None]
    while Q.qsize() > 0:

        current = Q.get()

        if current is None:
            Q.put(None)
            h += 1
            level_best.append(1e10)
            level_best_solvers.append(None)
            if h >= height:
                break
            continue

        avail = available - set(current.seq)
        for s in avail:
            total += 1
            sol = copy.copy(current)
            if len(sol.states) > 1:
                sol.pop_state()
            print(f"{sol.seq + [s]}")

            sol.next(s)
            sol.next(0)
            last_state = sol.get_last_state()
            if not last_state or len(last_state.schedule) == 0:
                infeasible += 1
                continue
            path_distance = last_state.distances[0]
            if path_distance < level_best[h]:
                level_best[h] = path_distance
                level_best_solvers[h] = sol

            if h <= height:
                Q.put(sol)

        print(f"Completed {current.seq}\n")

    return level_best, level_best_solvers



if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)
    root = DPSolver(P, seq=[])
    best_dist, best_solv = exhaustive(root=root, available=set(ALPHA), problem=P, height=3)
