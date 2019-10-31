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


def sliding_window_solver(k, n, solver):
    levels = []
    finished = False
    mode = 'grow'

    solver = copy.deepcopy(solver)
    while not finished:

        if mode == 'grow':
            for i in range(3):
                sliding_window()
    #         for i in range(3):
    #             _s = sliding_window(i, i + 1, S)  # Pass in the seq to the sliding window
    #             if _s and b[0] != _s.seq and tuple(_s.seq) not in level_sequences:
    #                 Q.append(_s)
    #                 level_sequences.add(tuple(_s.seq))
    #
    #         pass
    #     elif mode == 'shrink':
    #         pass
    #     else:
    #         print("finished")


if __name__ == "__main__":
    from dsp.Problem import Problem, load_problem
    from dsp.Solver import DPSolver
    from dsp.Runner import SequenceSolver
    import time
    import pprint as pp

    # Create sequence solver object
    P = load_problem()

    start = time.time()
    seq = [0, 32, 63, 12, 0]  # Should yield [0, 32, 63, 4, 0] as optimal
    solver = solve_sequence(problem=P, seq=seq)
    # seq2 = [0, 32, 30, 63, 4, 0]
    # seq2 = [0, 8, 5, 30, 63, 4, 0]
    # seq = [0, 33, 15, 8, 5, 44, 26, 56, 53, 38, 4, 32, 30, 63, 12, 43, 7, 16, 23, 61, 28, 0]
    # seq = [0, 4, 0]
    result = sliding_window(2, 1, solver)
    print(result)
