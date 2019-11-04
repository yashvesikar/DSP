import copy
import time
from collections import deque

import numpy as np

from dsp.Window import sliding_window
from dsp.Model import Optimizer
from dsp.Problem import Problem, load_problem
from dsp.Solver import DPSolver, solve_sequence
from dsp.Truncation import select_truncation


ALPHA = np.arange(1, 64)
np.random.seed(10)

def repopulate_queue(Q, solutions):
    rand = np.random.permutation(len(solutions))
    old_pop = [solutions[rand[i]] for i in range(10)]
    problem = load_problem()
    for s in old_pop:
        if s:
            solver = solve_sequence(problem, s[-1])
            if solver.feasible:
                res = sliding_window(2, 1, solver, return_first=False, restrict_available=False)
                if res.feasible:
                    Q.append(res)




class HeuristicTreeSolver:
    def __init__(self, **kwargs):
        # super().__init__(**kwargs)
        self.problem = kwargs.get('problem')
        self.root = DPSolver(self.problem, seq=[])
        self.max_depth = kwargs.get('max_depth')
        self.available = set(self.problem.ships_in_working_area())
        self.truncation_args = kwargs.get('truncation_args')

        # Solution Trackers
        self.best_distances = []
        self.best_solvers = []

        # Final results
        self.results = {}

    def _next(self, current, Q, depth):
        """
        Solves 1 sequence level
        :param current: Current solver to expand
        :param Q: Current Solver Queue
        :param depth: Current depth in tree
        :return: Updated Queue, best distance and best solver from this sequence expansion
        """
        best_distance = 1e10
        best_solver = None

        avail = self.available - set(current.seq)

        for s in avail:

            # Solution generation and evaluation
            sol = copy.copy(current)
            if len(sol.states) > 1:
                sol.pop_state()

            if sol.next(s) is False:
                continue
            if sol.next(0) is False:
                continue

            # Feasibility checks
            last_state = sol.last_state()
            if not last_state or len(last_state) == 0:
                continue

            # Feasible Solution updates
            result = sol.get_result()
            sol.result = result

            # Update counters and trackers
            if result.distance < best_distance:
                best_distance = result.distance
                best_solver = sol

            if depth <= self.max_depth:
                Q.append(sol)
        return Q, best_distance, best_solver

    def solve(self, verbose=False):
        """

        :return:
        """

        # Set the harbor
        self.root.next(0)
        depth = 0  # Height counter
        Q = deque([self.root, None])

        # ------------------------------------- Submodules -----------------------------------
        if select_truncation(self.truncation_args.get('method')):
            truncation = select_truncation(self.truncation_args.get('method'))
        elif truncation_args.get('method') is None:
            truncation = select_truncation('exhaustive')
        else:
            raise BaseException(f"No truncation submodule: {truncation_args.get('method')}")
        # ------------------------------------------------------------------------------------

        # -------------------------------- Solution trackers ----------------------------
        # All feasible
        everything = []
        selected = []
        all_seq = []
        best_distance = 1e10
        best_solver = None
        # ------------------------------------------------------------------------------------
        while len(Q) > 0:

            current = Q.popleft()

            # -------------------------------- Level processing ----------------------------
            if current is None:
                if len(Q) == 0:
                    # repopulate_queue(Q=Q, solutions=everything[-1])
                    break
                # Truncation
                # everything.append([l.solution for l in Q if l])
                # all_seq.append([l.seq for l in Q if l])
                Q, data = truncation(Q=Q, **self.truncation_args)

                selected.append([l.result for l in Q if l])
                if verbose:
                    print(f"Finished Processing Level # {depth + 1} - {best_distance} - {best_solver.seq}")

                # Update trackers
                depth += 1
                self.best_distances.append(best_distance)
                best_distance = 1e10
                self.best_solvers.append(best_solver)
                if depth >= self.max_depth:
                    break
                continue
            # -------------------------------- Exploration & Evaluation --------------------------
            Q, bd, bs = self._next(current=current, Q=Q, depth=depth)
            if bd <= best_distance:
                best_distance = bd
                best_solver = bs
            # ------------------------------------------------------------------------------------

        result = {}
        result["results"] = self.best_distances
        result["solvers"] = self.best_solvers
        result["selected"] = selected
        return result


if __name__ == "__main__":
    # Create sequence solver object
    P = load_problem(T=6)

    truncation_args = {'limit': 1000, 'method': "distance"}
    SeqSolver = HeuristicTreeSolver(problem=P, max_depth=10, truncation_args=truncation_args)

    # truncation_args = {'limit': 1000, 'method': "decomposition", 'w': 0.306}

    start = time.time()

    result = SeqSolver.solve()
    end = time.time()
    print(f"{end - start}")
