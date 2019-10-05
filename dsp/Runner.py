import copy
import time
import threading

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

    best_dist = [0]
    level_best_dist = 1e10

    best_solver = [root]
    level_best_solver = None
    q_size = [1]
    while Q.qsize() > 0:

        current = Q.get()

        if current is None:
            Q.put(None)
            h += 1
            q_size.append(Q.qsize())
            best_dist.append(level_best_dist)
            level_best_dist = 1e10
            best_solver.append(level_best_solver)

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

            if sol.next(s) is False: continue
            if sol.next(0) is False: continue

            last_state = sol.get_last_state()
            if not last_state or len(last_state) == 0:
                infeasible += 1
                continue

            path_distance = last_state.distances[0]
            if path_distance < level_best_dist:
                level_best_dist = path_distance
                level_best_solver = sol

            if h <= height:
                Q.put(sol)
    print(infeasible)
    return best_dist, best_solver, q_size


if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)
    root = DPSolver(P, seq=[])
    start = time.time()
    best_dist, best_solv, q_size = exhaustive(root=root, available=set(ALPHA), problem=P, height=2)
    end = time.time()
    print(f"{end-start}")


    # sched = best_solv[-1].construct_from_states(best_solv[-1].states)[0]
    # seq = best_solv[-1].seq
