import itertools

import numpy as np
from dsp.Problem import Problem
from dsp.truncation import exhsaustive_truncation, nds_truncation,distance_truncation, time_truncation
from dsp.Solver import DPSolver
from dsp.Runner import SequenceSolver
import time
import pprint as pp


def sliding_window(k, n, seq, sched):
    """

    :param k: Number of ships to expand by
    :param n: Number of time windows to expand into
    :param seq:
    :param sched:
    :return:
    """
    success = 0
    fail = 0
    S = DPSolver(problem=P, seq=[])
    best = 1e10
    best_solver = None

    for i in range(len(seq)-k):
        avail = P.ships_in_working_area(start=sched[i], end=sched[i + n])
        avail = set(avail) - set(seq[:i+1] + seq[i+n:])
        if 0 in avail:
            avail.remove(0)

        sub_seq = list(itertools.permutations(avail, k))


        for ship in sub_seq:
            _seq = seq[:i+1] + list(ship) + seq[i+n:]

            S.clear()

            result = S.solve(seq=_seq, return_distance=True)
            if result:
                print(f"Success: {_seq}, dist: {result[1]}")
                success += 1
                s, d = result
                if d < best:
                    best = d
                    best_solver = (_seq, s, d)
            else:
                # print(f"Failed: {new_seq}")
                fail += 1

    print(f"Success: {success}, Failure: {fail}")
    print(best_solver)
    return best_solver


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
    SeqSolver = SequenceSolver(problem=P, root=root, height_limit=3)

    # Create values for sequence search
    ALPHA = set(P.in_working_area)
    truncation_args = {'limit': 1000, 'type': 'alpha'}
    exploration = None

    S = DPSolver(problem=P, seq=[])

    start = time.time()

    seq = [0, 32, 63, 4, 0]
    seq2 = [0, 32, 30, 63, 4, 0]
    # seq2 = [0, 8, 5, 30, 63, 4, 0]

    sched1, dist1 = S.solve(seq=seq, return_distance=True)
    sched1[-1] = P.m
    print(f"Length {len(sched1)-2} dist: {dist1}")
    S.clear()
    sched2, dist2 = S.solve(seq=seq2, return_distance=True)
    print(f"Length {len(sched2)-2} dist: {dist2}")

    results = []
    s = sched1
    for i in range(10):
        res = sliding_window(2, 2, seq, sched=s)
        if res:
            results.append(res)
            seq, s, d = res
            s[-1] = P.m
    pp.pprint(results)