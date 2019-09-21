import math

import numpy as np
from scipy.spatial.distance import cdist

from DP2 import Problem


class DPSolver:
    def __init__(self, problem):
        self.problem = problem

    @staticmethod
    def distance(n1, n2):
        # return np.sqrt(np.sum((n2 - n1) ** 2))
        return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)

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

        Positions = np.zeros((1, 2))
        Times = [0]
        Path = [[[0, 0]]]
        Schedule = [[0]]

        for k in range(len(ships) - 1):


            s1 = ships[k]
            s2 = ships[k + 1]

            time2, points2 = s2.get_times(array=True), s2.get_positions()

            # Returning to the harbor
            if ships[k + 1].id == 0:
                time2, points2 = np.array([time2[-1]]), np.array([points2[-1]])

            distance_matrix = cdist(points2, Positions)
            if distance_matrix.shape[1] == 1:
                axis=0
            else:
                axis=1

            decision_matrix = np.argsort(distance_matrix, axis=axis)

            _positions = []
            _times = []
            _path = []
            _schedule = []

            for j in range(decision_matrix.shape[0]):

                for i in decision_matrix[j]:

                    if axis == 0:
                        j, i = i, 0
                        # travel = self.travel_time(distance_matrix[i, j])
                    # else:
                        # travel = self.travel_time(distance_matrix[j, i])

                    travel = self.travel_time(distance_matrix[j, i])

                    next_time = Times[i] + travel + 0.6

                    if next_time <= time2[j]:
                        _positions.append(points2[j])
                        _times.append(time2[j])
                        _path.append(Path[i] + [points2[j].tolist()])
                        _schedule.append(Schedule[i] + [time2[j]])
                        break

            Path = _path
            Positions = _positions
            Schedule = _schedule
            if _times[0] != np.inf:
                Times = _times

        return Path[0], Schedule[0]


if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("data/x.csv", delimiter=",")
    y_data = np.genfromtxt("data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)

    S = DPSolver(P)

    # seq = [8, 5, 30, 63, 4]
    seq = [56, 26, 33, 8, 12]

    pos, sched = S.solve_sequence(seq)

    d = 0
    for p in range(len(pos) - 1):
        d += S.distance(pos[p + 1], pos[p])
    print(f"TOTAL DISTANCE: {d}")
