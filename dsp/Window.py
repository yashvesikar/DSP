import itertools
import numpy as np
import copy
from dsp.Solver import DPSolver, solve_sequence


def sliding_window(k, n, S):
    """

    :param k: Number of ships to expand by
    :param n: Number of ships to remove
    :param S: DPSolver instance
    :return:
    """
    success = 0
    fail = 0
    seq, sched = S.seq, S.schedule
    P = S.problem
    best = 1e10
    best_solver = None
    best_solver_obj = None
    n += 1
    for i in range(len(seq)-n):
        avail = P.ships_in_working_area(start=sched[i], end=sched[i + n])
        avail = set(avail) - set(seq[:i+1] + seq[i+n:])
        if 0 in avail:
            avail.remove(0)

        sub_seq = list(itertools.permutations(avail, k))


        for ship in sub_seq:
            _seq = seq[:i+1] + list(ship) + seq[i+n:]

            # S.clear()
            result = solve_sequence(problem=P, seq=_seq)
            # result = S.solve(seq=_seq, return_distance=True)
            if result:
                # print(f"Success: {_seq}, dist: {result[1]}")
                success += 1
                s, d = result.schedule, result.dist
                if d < best:
                    best = d
                    best_solver = (_seq, s, d)
                    best_solver_obj = copy.deepcopy(result)
                    # best_solver = (_seq, s, d)
            else:
                # print(f"Failed: {new_seq}")
                fail += 1

    return best_solver_obj


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
    from dsp.Problem import Problem
    from dsp.Solver import DPSolver
    from dsp.Runner import SequenceSolver
    import time
    import pprint as pp

    # Time frame
    T = 6

    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    # Create sequence solver object
    P = Problem(xy_data, T=T)
    root = DPSolver(P, seq=[])
    SeqSolver = SequenceSolver(problem=P, root=root, height_limit=3)

    # Create values for sequence search
    ALPHA = set(P.in_working_area)
    truncation_args = {'limit': 1000, 'type': 'alpha'}
    exploration = None

    S = DPSolver(problem=P, seq=seq)

    start = time.time()

    seq = [0, 32, 63, 4, 0]
    # seq2 = [0, 32, 30, 63, 4, 0]
    # seq2 = [0, 8, 5, 30, 63, 4, 0]
    # seq = [0, 33, 15, 8, 5, 44, 26, 56, 53, 38, 4, 32, 30, 63, 12, 43, 7, 16, 23, 61, 28, 0]
    # seq = [0, 4, 0]
    sched1, dist1 = S.solve(return_distance=True)
    sched1[-1] = P.m
    print(f"Length {len(sched1)-2} dist: {dist1}")
    S.clear()

    results = []
    s = sched1
    # for i in range(20):
    res = sliding_window(1, 2, seq, sched=s)
    results.append(res)

        # if res:
        #     seq, s, d = res
        #     s[-1] = P.m
        # else:
        #     break
    # pp.pprint(results)
