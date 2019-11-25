import os

import numpy as np

from pymoo.model.problem import Problem

# from dsp.Solver import solve_sequence

W = 5 / 60
class Ship:
    def __init__(self, number, positions, times):
        self.id = number
        self.positions = positions
        self.times = times

    def get_positions(self, range=None):
        if range:
            start, end = range
            return self.positions[start - self.times[0]: end+1 - self.times[0]]
        return self.positions

    def get_position(self, time):
        if self.exists(time):
            return self.positions[time - self.times[0]]

    def get_times(self, array=False):
        if not self.times:
            return np.array([])
        if array and self.times:
            return np.arange(self.times[0], self.times[1])
        return self.times

    def get_relative_times(self, array=False):
        if not self.times:
            return np.array([])
        if array and self.times:
            return np.arange(self.times[0] - self.times[0], self.times[1] - self.times[0])
        return self.times - self.times[0]



    def exists(self, time):
        return self.times[0] <= time < self.times[1]


class ShipProblem(Problem):
    def __init__(self, xy_data, T=6, pf=None, **kwargs):
        self.data = xy_data
        self.T = T
        self.ship_data = None
        self.pf = pf

        self.m = np.floor((T / W) + 0.1)

        self.construct(xy_data, T)

        self.in_working_area = set(self.ships_in_working_area()[1:])
        self.n_avail = len(self.in_working_area)
        super().__init__(**kwargs)

    def get_ships_positions(self, S):
        """

        :param S: A list of ships for which to return all positions
        :return:
        """
        pos = []
        for s in S:
            pos.append(self.ship_data[s].get_positions())
        return pos

    def ships_in_working_area(self, start=0, end=None):
        if end is None:
            end = self.m

        w = []

        for s in self.ship_data:
            # Time entering WA > end of window
            # Or Time leaving WA < start of window
            if s.times[0] > end or s.times[1] < start:
                continue
            else:
                w.append(s.id)
        return w


    def construct(self, xy_data, T):
        ships = []

        # Size of the time slot
        w = 5/60

        # number of ships in the working area in time [0, T]
        n = xy_data.shape[1]

        # Number of time slots in time [0, T]
        m = int(np.floor(T / w + 0.1))

        # For loop to fill out first, and last array
        for s in range(n):
            c = np.all(xy_data[:, s] != 0, axis=1)
            non_zero = np.where(c)[0]

            if len(non_zero):
                first = non_zero[0]
                last = non_zero[-1] + 1
            elif s == 0:
                first = 0
                last = m + 1
            else:
                first = -1
                last = -1

            ships.append(Ship(number=s, positions=xy_data[first:last, s], times=(first, last)))

        self.ship_data = ships


    def get_ship(self, s):
        return self.ship_data[s]


class MOOProblem(ShipProblem):
    def __init__(self, xy_data, T=6, **kwargs):

        super().__init__(xy_data=xy_data,
                         T=T,
                         n_var=1,
                         n_obj=2,
                         n_constr=1,
                         elementwise_evaluation=True,
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        val = x[0]
        seq = [0] + val + [0]

        solver = solve_sequence(self, seq)
        out["F"] = np.array([- float(len(val)), solver.result.distance])
        out["G"] = 0.0 if solver.result.feasible else 1.0
        out["solver"] = solver

    def _calc_pareto_front(self, *args, **kwargs):
        pf = self.pf.astype(np.float) * [-1, 1]
        return pf


def load_data(pf=False, T=6):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Data
    x_data = np.genfromtxt(os.path.join(root, "data", "x.csv"), delimiter=",")
    y_data = np.genfromtxt(os.path.join(root, "data", "y.csv"), delimiter=",")
    xy_data = np.stack([x_data, y_data], axis=2)

    if pf and T:
        _pf = os.path.join(root, "data", "pf", f"{T}.csv")
        if os.path.exists(_pf):
            _pf =  np.loadtxt(_pf)
            return (xy_data, _pf)
        else:
            print(f"{_pf} Does not exist.")

    return xy_data


def load_problem(problem='ship', T=6, **kwargs):
    data, pf = load_data(pf=True, T=T)

    P = ShipProblem(xy_data=data, pf=pf, T=T)

    return P

if __name__ == "__main__":
    P = load_problem(6)
