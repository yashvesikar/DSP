import copy
import os
import pickle
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dsp.Problem import Problem, load_problem
from dsp.Solver import DPSolver, solve_sequence
from dsp.Exploration import select_exploration
from dsp.Truncation import select_truncation
from dsp.Window import sliding_window

ALPHA = np.arange(1, 64)
np.random.seed(10)


class SequenceSolver:
    def __init__(self, problem, height_limit=5):
        self.P = problem
        self.root = DPSolver(self.P, seq=[])
        self.height_limit = height_limit
        self.available = self.calculate_available()

        # Final results
        self.results = {}

    def calculate_available(self):
        T = self.P.T
        available = set(np.nonzero(self.P.data[:T, :, 0])[1])
        return available

    def sequence_search(self, available, truncation_args=None):
        """

        :param available:
        :param truncation_args:
        :return:
        """

        # Set the harbor
        self.root.next(0)
        h = 0  # Height counter
        Q = deque([self.root, None])
        infeasible = 0
        total = 0

        # For plotting

        # ------------------------------------- Submodules -----------------------------------
        if select_truncation(truncation_args.get('method')):
            truncation = select_truncation(truncation_args.get('method'))
        elif truncation_args.get('method') is None:
            truncation = select_truncation(truncation_args.get('exhaustive'))
        else:
            raise BaseException(f"No truncation submodule: {truncation_args.get('method')}")
        # ------------------------------------------------------------------------------------

        # -------------------------------- Solution trackers ----------------------------
        # All feasible
        everything = []
        selected = []
        all_seq = []

        # Overall Best
        best_dist = [0]
        best_solver = [self.root]

        # Best on current level
        level_best_dist = 1e10
        level_best_solver = None
        # ------------------------------------------------------------------------------------
        while len(Q) > 0:

            current = Q.popleft()

            # -------------------------------- Level processing ----------------------------
            if current is None:
                if len(Q) == 0:
                    break

                # Truncation
                everything.append([l.solution for l in Q if l])
                all_seq.append([l.seq for l in Q if l])
                Q, data = truncation(Q=Q, **truncation_args)

                selected.append([l.solution for l in Q if l])
                print(f"Finished Processing Level # {h + 1}")
                # Update trackers
                h += 1
                best_dist.append(level_best_dist)
                level_best_dist = 1e10
                best_solver.append(level_best_solver)
                if h >= self.height_limit:
                    break
                continue
            # ------------------------------------------------------------------------------------

            # -------------------------------- Exploration & Evaluation ----------------------------
            avail = available - set(current.seq)

            for s in avail:
                total += 1

                # Solution generation and evaluation
                sol = copy.copy(current)
                if len(sol.states) > 1:
                    sol.pop_state()

                if sol.next(s) is False: continue
                if sol.next(0) is False: continue

                # Feasibility checks
                last_state = sol.get_last_state()
                if not last_state or len(last_state) == 0:
                    infeasible += 1
                    continue

                # Feasible Solution updates
                sol.update(feasible=True)  # Set the DP solver to feasible
                sol.solution = (sol.dist, len(sol.states) - 2, sol.schedule)

                # Update counters and trackers
                if sol.dist < level_best_dist:
                    level_best_dist = sol.dist
                    level_best_solver = sol

                if h <= self.height_limit:
                    Q.append(sol)
            # ------------------------------------------------------------------------------------

        print("Finished")
        result = {}
        result["infeasible"] = infeasible
        result["everything"] = everything
        result["selected"] = selected
        result["best_dist"] = best_dist
        result["best_solver"] = best_solver
        result["total_evaluations"] = total
        return result


if __name__ == "__main__":
    # Create sequence solver object
    P = load_problem(T=6)
    ALPHA = set(P.ships_in_working_area())

    SeqSolver = SequenceSolver(problem=P, height_limit=4)

    # truncation_args = {'limit': 1000, 'method': "decomposition", 'w': 0.306}
    truncation_args = {'limit': 100, 'method': "distance"}

    start = time.time()

    result = SeqSolver.sequence_search(available=ALPHA, truncation_args=truncation_args)
    end = time.time()
    print(f"{end - start}")
