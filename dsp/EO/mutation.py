import itertools

import numpy as np
from pymoo.model.mutation import Mutation

from dsp.Problem import load_problem
from dsp.Solver import solve_sequence


class MOOMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):

        X = X.astype(np.float)

        for k, off in enumerate(X):
            sched = off.schedule
            windows = []
            for i in range(len(sched) - 1):
                windows.append(sched[i + 1] - sched[i])
            ind = np.argmax(windows)
            _start = problem.get_ship(off.seq[ind - 1]).times[0]
            _end = problem.get_ship(off.seq[ind + 1]).times[-1]

            avail = set(problem.data.ships_in_working_area(start=_start, end=_end)) - set(
                off.seq[:ind] + off.seq[ind + 1:])

            choice = np.random.randint(0, 3)
            if choice == 0:
                # Grow
                sub_seq = itertools.permutations(avail, 2)

                sub_seq = [s for s in sub_seq if len(set(seq[ind]) - set(s)) == 0]

                X[k] = off.seq[:ind] + sub_seq[np.random.randint(0, len(sub_seq))] + off.seq[ind + 1:]

            elif choice == 1:
                # Shrink
                X[k] = off.seq[:ind] + off.seq[ind + 1:]
            else:
                # Permute
                sub_seq = set(itertools.permutations(avail, 1)) - set(off.seq[ind])

                X[k] = off.seq[:ind] + sub_seq[np.random.randint(0, len(sub_seq))] + off.seq[ind + 1:]

        return X
