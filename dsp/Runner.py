import copy
import os
import pickle
import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from dsp.Problem import Problem
from dsp.Solver import DPSolver
from dsp.Exploration import select_exploration
from dsp.Truncation import select_truncation
from dsp.Window import sliding_window

ALPHA = np.arange(1, 64)
np.random.seed(10)


class SequenceSolver:
    def __init__(self, root, problem, height_limit=5):
        self.P = problem
        self.root = root
        self.height_limit = height_limit
        self.available = self.calculate_available()

        # Final results
        self.results = {}

    def calculate_available(self):
        T = self.P.T
        available = set(np.nonzero(self.P.data[:T, :, 0])[1])
        return available

    def sequence_search(self,
                        available,
                        truncation_args=None,
                        exploration_args=None,
                        save=None):
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

        # For plotting

        # ------------------------------------- Submodules -----------------------------------
        if select_truncation(truncation_args.get('method')):
            truncation = select_truncation(truncation_args.get('method'))
        elif truncation_args.get('method') is None:
            truncation = select_truncation(truncation_args.get('exhaustive'))
        else:
            raise BaseException(f"No truncation submodule: {truncation_args.get('method')}")

        if exploration_args and select_exploration(exploration_args.get('method')):
            exploration = select_truncation(truncation_args.get('method'))
        else:
            exploration = None
        # else:
        #     raise BaseException(f"No exploration submodule: {exploration_args.get('method')}")
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
                    if len(all_seq):

                        previous_level = all_seq[-1]
                        b = previous_level[:10]
                        level_sequences = set()
                        for s in b:
                            S = DPSolver(P, seq=[])
                            S.solve(s)
                            for i in range(3):
                                _s = sliding_window(i, i+1, S)   # Pass in the seq to the sliding window
                                if _s and b[0] != _s.seq and tuple(_s.seq) not in level_sequences:
                                    Q.append(_s)
                                    level_sequences.add(tuple(_s.seq))

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
            if exploration:
                if current is not self.root:
                    avail = exploration(avail=avail, current=current, problem=self.P)

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
                sol.solution = (sol.dist, len(sol.states) - 2, sol.schedule[-1])

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
        # if save and os.path.isdir(save):
        #     # fname = os.path.basename(os.path.normpath(save))
        #     with open(save, 'wb') as fp:
        #         pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if False:

            everything = [item for sublist in everything for item in sublist]
            selected = [item for sublist in selected for item in sublist]

            d = [l.solution[0] for l in everything if l]
            a = [l.solution[1] for l in everything if l]
            t = [l.solution[2] for l in everything if l]

            d2 = [l.solution[0] for l in selected if l]
            a2 = [l.solution[1] for l in selected if l]
            t2 = [l.solution[2] for l in selected if l]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(a, t, d, c='b', marker='.')
            ax.scatter(a2, t2, d2, c='r', marker='.', alpha=0.5)

            ax.set_xlabel('Alpha')
            ax.set_ylabel('Time')
            ax.set_zlabel('Distance')
            plt.title(f"Exhaustive level {len(Q[0].states) - 2} - count {len(Q)} - {truncation_args.get('type')}")

            plt.show()
        return result


if __name__ == "__main__":

    # Time frame
    T = 6

    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    # Create sequence solver object
    P = Problem(xy_data, T=T)
    root = DPSolver(P, seq=[])
    ALPHA = set(P.in_working_area)

    SeqSolver = SequenceSolver(problem=P, root=root, height_limit=20)

    # Create values for sequence search

    # truncation_args = {'limit': 1000, 'method': "decomposition", 'w': 0.306}
    truncation_args = {'limit': 10, 'method': "distance"}
    exploration = None

    start = time.time()
    method = 0
    if method == 0:
            result = SeqSolver.sequence_search(available=ALPHA,
                                           truncation_args=truncation_args)
    end = time.time()
    print(f"{end - start}")
