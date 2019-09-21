import numpy as np
from scipy.spatial.distance import cdist

from DP2 import Problem


class DPSolver:
    def __init__(self, problem):
        self.problem = problem

    @staticmethod
    def distance(n1, n2):
        return np.sqrt(np.sum((n2 - n1) ** 2))
        # return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)

    @staticmethod
    def travel_time(d):
        v = 46.3  # in km/h
        t = d / v

        w = 5 / 60

        return np.floor(t / (w + 0.1))

    def solve_sequence(self, seq):
        """

        :param seq:
        :return:
        """
        assert len(seq) > 0

        if seq[0] != 0:
            seq.insert(0, 0)
        if seq[-1] != 0:
            seq.append(0)

        ships = []
        for s in seq:
            ships.append(self.problem.get_ship(s))

        P = np.zeros((1, 2))
        T = [0]
        path = [[[0, 0]]]
        sched = [[0]]

        for k in range(len(ships) - 1):

            s1 = ships[k]
            s2 = ships[k + 1]

            time2, points2 = s2.get_times(array=True), s2.get_positions()

            D = cdist(points2, P)
            X = np.argsort(D, axis=1)

            _p = []
            _t = []
            _path = []
            _sched = []

            for j in range(X.shape[0]):

                for i in X[j]:

                    next_time = T[i] + self.travel_time(D[j, i]) + 0.6

                    if next_time <= time2[j]:
                        _p.append(points2[j])
                        _t.append(time2[j])
                        _path.append(path[i] + [points2[j].tolist()])
                        _sched.append(sched[i] + [time2[j]])
                        break

            path = _path
            P = _p
            sched = _sched
            if _t[0] != np.inf:
                T = _t

        return path, sched


if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("data/x.csv", delimiter=",")
    y_data = np.genfromtxt("data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)

    S = DPSolver(P)

    seq = [8, 5, 30, 63, 4]

    pos, sched = S.solve_sequence(seq)

    d = 0
    for p in range(len(pos) - 1):
        d += S.distance(pos[p + 1], pos[p])
    print(f"TOTAL DISTANCE: {d}")
