import numpy as np

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


class Problem:
    def __init__(self, xy_data, T=6):
        self.ships = None
        self.data = xy_data
        w = 5/60
        self.m = np.floor((T/w) + 0.1)
        self.T = T
        self.construct(xy_data, T)
        self.in_working_area = self.ships_in_working_area()

    def get_ships_positions(self, S):
        """

        :param S: A list of ships for which to return all positions
        :return:
        """
        pos = []
        for s in S:
            pos.append(self.ships[s].get_positions())
        return pos

    def ships_in_working_area(self, start=0, end=None):
        if end is None:
            end = self.m

        w = []

        for s in self.ships:
            if start <= s.times[0] <= end:
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

        # first = np.full(n, m, dtype=np.int)  # First slot when ship i is in the work area
        first = m  # First slot when ship i is in the work area
        # last = np.full(n, 0, dtype=np.int)  # Last slot when ship i is in the work area
        last = 0  # Last slot when ship i is in the work area

        # For loop to fill out first, and last array
        for s in range(n):

            c = np.all(xy_data[:, s] != 0, axis=1)
            non_zero = np.where(c)[0]
            # number = s
            if len(non_zero):
                # first[s] = non_zero[0]
                first = non_zero[0]
                # last[s] = non_zero[-1] + 1
                last = non_zero[-1] + 1
            elif s == 0:
                # first[s] = 0
                first = 0
                # last[s] = m
                last = m + 1
                # number = 0
            else:
                # first[s] = -1
                first = -1
                # last[s] = -1
                last = -1
            ships.append(Ship(number=s, positions=xy_data[first:last, s], times=(first, last)))

        self.ships = ships
        # # Harbor should always be (0, 0)
        # # Replace all other (0, 0) values with nan
        # xy_data[:m, 1:][xy_data[:m, 1:] == np.array([0, 0])] = np.nan
        # self.positions = xy_data
        # self.times = np.column_stack((first, last))
        print()

    def get_ship(self, s):
        return self.ships[s]


def load_problem(T=6):
    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data, T=T)
    return P

if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("data/x.csv", delimiter=",")
    y_data = np.genfromtxt("data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)
