import itertools
import numpy as np
import copy
from dsp.Solver import DPSolver, solve_sequence



def sliding_window(k, n, S, return_first=False):
    """

    :param k: Number of ships to expand by
    :param n: Number of ships to remove
    :param S: DPSolver instance
    :return:
    """
    success = 0
    fail = 0
    seq, sched, problem = S.seq, S.schedule, S.problem
    best_dist = 1e10
    best_solver = None
    n += 1
    for i in range(len(seq)-n):
        avail = problem.ships_in_working_area(start=sched[i], end=sched[i + n])
        avail = set(avail) - set(seq[:i+1] + seq[i+n:])
        if 0 in avail:
            avail.remove(0)

        sub_seq = list(itertools.permutations(avail, k))

        for ship in sub_seq:
            _seq = seq[:i+1] + list(ship) + seq[i+n:]
            result = solve_sequence(problem=problem, seq=_seq)
            if result.feasible:
                success += 1

                # Return the first feasible solution
                if return_first:
                    return result

                if result.dist < best_dist:
                    best_dist = result.dist
                    best_solver = result

            else:
                fail += 1

    return best_solver


def ping_pong_solver(solvers):

    finished = False
    mode = 'opt'
    solver_ind = 0
    solver = solvers[solver_ind]
    levels = [s.dist for s in solvers]
    improved = False
    while not finished:

        if mode == 'grow':
            res = sliding_window(2, 1, solver)
            if res.dist < levels[len(res.seq)-2]:
                levels[len(res.seq)-2] = res.dist
                solvers[len(res.seq) - 2] = res
                improved = True
            if improved:
                mode = 'opt'

        # elif mode == 'shrink':
        #     res = sliding_window(1, 2, solver)

        elif mode == 'opt':
            for i in range(min(len(solver.seq)-2, 3)):
                res = sliding_window(i, i, solver)
                if res.dist < levels[len(res.seq)-3]:
                    levels[len(res.seq) - 2] = res.dist
                    solvers[len(res.seq) - 2] = res
                    improved = True

            if not improved:
                mode = 'grow'
                solver_ind += 1
                solver = solvers[solver_ind]
            else:
                improved = False


if __name__ == "__main__":
    from dsp.Problem import Problem, load_problem
    from dsp.Solver import DPSolver
    from dsp.Runner import SequenceSolver
    import time
    import pprint as pp

    # Create sequence solver object
    P = load_problem()
    truncation_args = {'limit': 100, 'method': "distance"}
    ALPHA = set(P.ships_in_working_area())
    SeqSolver = SequenceSolver(problem=P, height_limit=15)
    result = SeqSolver.sequence_search(available=ALPHA, truncation_args=truncation_args)

    result = ping_pong_solver(result["best_solver"])
    print(result)
