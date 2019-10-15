import copy
import time
from scipy.spatial.distance import cdist, pdist

import numpy as np

from dsp.Problem import Problem
from dsp.Solver import DPSolver
from dsp.exploration import distance_based
from dsp.truncation import distance_truncation, nds_truncation
from collections import deque

ALPHA = np.arange(1, 64)
np.random.seed(10)


class SequenceSolver:
    def __init__(self, root, problem, height_limit=5, level_size_limit=10000):
        self.P = problem
        self.root = root
        self.level_limit = level_size_limit
        self.height_limit = height_limit
        self.available = self.calculate_available()
        self.results = {}

    def calculate_available(self):
        T = self.P.T
        available = set(np.nonzero(self.P.data[:T, :, 0])[1])
        return available

    def sequence_search(self,
                        available,
                        truncation=None,
                        exploration=None,
                        truncation_args=None):
        """

        :param truncation_args:
        :param exploration:
        :param truncation:
        :param problem:
        :param available: Set of available ship ids
        :param visited: set of visited ship ids
        :return:
        """

        # Set the harbor
        self.root.next(0)
        h = 0  # Height counter
        Q = deque([self.root, None])
        infeasible = 0
        total = 0

        postprocessing = []
        best_dist = [0]
        level_best_dist = 1e10

        best_solver = [self.root]
        level_best_solver = None
        q_size = [1]
        while len(Q) > 0:

            current = Q.popleft()

            if current is None:

                Q, data = truncation(Q=Q, **truncation_args)

                print(f"Finished Processing Level # {h}")
                h += 1
                q_size.append(len(Q))
                best_dist.append(level_best_dist)
                level_best_dist = 1e10
                best_solver.append(level_best_solver)
                postprocessing.append(data)
                if h >= self.height_limit:
                    break
                continue

            avail = available - set(current.seq)
            if exploration:
                if current is not self.root:
                    avail = exploration(avail=avail, current=current, problem=self.P)

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
                sol.solution = (last_state.distances[0], len(sol.states)-2, last_state.times[0])
                path_distance = last_state.distances[0]
                if path_distance < level_best_dist:
                    level_best_dist = path_distance
                    level_best_solver = sol
                if h <= self.height_limit:
                    Q.append(sol)
        print(f"Infeasible Count: {infeasible}")

        result = {}
        result["best_dist"] = best_dist
        result["best_solver"] = best_solver
        result["q_size"] = q_size
        result["total_evaluations"] = total
        result["postprocess"] = postprocessing
        return result


if __name__ == "__main__":

    # Time frame
    T = 6

    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    # Create sequence solver object
    P = Problem(xy_data, T=6)
    root = DPSolver(P, seq=[])
    SeqSolver = SequenceSolver(problem=P, root=root, height_limit=4)

    # Create values for sequence search
    ALPHA = set(P.in_working_area)
    truncation_args = {'limit': 10000}
    exploration = None

    start = time.time()
    result = SeqSolver.sequence_search(available=ALPHA,
                                       exploration=exploration,
                                       truncation=nds_truncation,
                                       truncation_args=truncation_args)
    end = time.time()
    print(f"{end - start}")
