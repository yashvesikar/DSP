import copy
import os
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from dsp.Problem import Problem
from dsp.Solver import DPSolver
from dsp.Exploration import select_exploration
from dsp.Truncation import select_truncation
from collections import deque

ALPHA = np.arange(1, 64)
np.random.seed(10)


class SequenceSolver:
    def __init__(self, root, problem, height_limit=5):
        self.P = problem
        self.root = root
        self.height_limit = height_limit
        self.available = self.calculate_available()

        # Algorithm intermediate data values
        self.max_t = 0
        self.max_d = 0

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

        #

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

        # Overall Best
        best_dist = [0]
        best_solver = [self.root]

        # Best on current level
        level_best_dist = 1e10
        level_best_solver = None
        # ------------------------------------------------------------------------------------
        pbar = tqdm(total=self.height_limit)
        while len(Q) > 0:

            current = Q.popleft()

            # Level processing
            if current is None:
                if len(Q) == 0:
                    break

                # Truncation
                # everything.append(Q)
                # truncation_args["max_d"] = self.max_d
                # truncation_args["max_t"] = self.max_t
                Q, data = truncation(Q=Q, **truncation_args)
                # selected.append(Q)
                # print(f"Finished Processing Level # {h + 1}")
                pbar.update(h)
                # Update trackers
                h += 1
                best_dist.append(level_best_dist)
                level_best_dist = 1e10
                best_solver.append(level_best_solver)
                if h >= self.height_limit:
                    break
                continue

            # Exploration
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
                sol.feasible = True  # Set the DP solver to feasible
                sol_sched = last_state.schedule[0]
                sol_distance = last_state.distances[0]
                sol.solution = (sol_distance, len(sol.states) - 2, sol_sched)

                # Update counters and trackers
                if sol_distance < level_best_dist:
                    level_best_dist = sol_distance
                    level_best_solver = sol
                if self.max_d < sol_distance:
                    self.max_d = sol_distance
                if self.max_t < sol_sched:
                    self.max_t = sol_sched

                if h <= self.height_limit:
                    Q.append(sol)
        # print(f"Infeasible Count: {infeasible}")
        print("Finished")
        result = {}
        result["infeasible"] = infeasible
        # result["everything"] = everything
        # result["selected"] = selected
        result["best_dist"] = best_dist
        result["best_solver"] = best_solver
        result["total_evaluations"] = total
        # if save and os.path.isdir(save):
        #     # fname = os.path.basename(os.path.normpath(save))
        #     with open(save, 'wb') as fp:
        #         pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pbar.close()
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
    SeqSolver = SequenceSolver(problem=P, root=root, height_limit=5)

    # Create values for sequence search
    ALPHA = set(P.in_working_area)
    # truncation_args = {'limit': 1000, 'method': "decomposition", 'w': 0.306}
    truncation_args = {'limit': 1000, 'method': "nds", 'w': 0.306, 'type': 'harbor'}
    exploration = None

    start = time.time()
    method = 0
    if method == 0:
        result = SeqSolver.sequence_search(available=ALPHA,
                                           truncation_args=truncation_args)
    # elif method == 1:
    #     result = SeqSolver.sequence_search(available=ALPHA,
    #                                        exploration=exploration,
    #                                        truncation=distance_truncation,
    #                                        truncation_args=truncation_args)
    # elif method == 2:
    #     result = SeqSolver.sequence_search(available=ALPHA,
    #                                        exploration=exploration,
    #                                        truncation=time_truncation,
    #                                        truncation_args=truncation_args)
    # elif method == 3:
    #     result = SeqSolver.sequence_search(available=ALPHA,
    #                                        exploration=exploration,
    #                                        truncation=exhsaustive_truncation,
    #                                        truncation_args=truncation_args)

    end = time.time()
    print(f"{end - start}")
