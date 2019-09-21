import math

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


def travel_time(d):
    v = 46.3  # in km/h
    t = d/v

    w = 5/60

    return np.floor(t/(w+0.1))

class ShipPath:
    def __init__(self, ID, positions):
        """
        Represents a Ship and its path through the work area
        :param ID: ID of the ship
        :param positions: List of position objects, each representing a position at a time
        """
        self.id = ID
        self.positions = positions
        if self.positions:
            self.time_range = (positions[0].time, positions[-1].time)

    def __repr__(self):
        return f"Ship {self.id}"

    def __getitem__(self, item):
        return self.positions[item]

    def __iter__(self):
        self.n = 0
        return iter(self.positions)

    def __next__(self):
        if self.n <= len(self.positions):
            item = self.positions[self.n]
            self.n += 1
            yield item

        else:
            raise StopIteration

    def get_positions(self):
        return self.positions


class Position:
    def __init__(self, pos, time, ship):
        """
        Class representing the position of a ship at time t
        :param pos: (x, y) coordinates of ship
        :param time: Time slot
        :param harbor: Whether this position is the harbor or not
        """
        self.pos = pos
        self.time = time
        self.ship = ship

    def __hash__(self):
        return hash((*self.pos, self.time))

    def __repr__(self):
        return f"Ship: {self.ship.id}, position: {self.pos}, time: {self.time}"

    def __getitem__(self, item):
        return self.pos[item]

    def get_coords(self):
        return self.pos

    def get_time(self):
        return self.time

    def get_id(self):
        return self.ship.id


class Problem:
    def __init__(self, xy_data):
        self.paths = self.construct(xy_data)
        self.time = 0

    def construct(self, xy_data):
        data = {}

        # number of ships in the working area in time [0, T]
        n = xy_data.shape[1]

        # Number of time slots in time [0, T]
        m = xy_data.shape[0]

        first = np.full(n, m, dtype=np.int)  # First slot when ship i is in the work area
        last = np.full(n, 0, dtype=np.int)  # Last slot when ship i is in the work area

        # For loop to fill out first, and last array
        for s in range(n):
            c = np.all(xy_data[:, s] != 0, axis=1)
            non_zero = np.where(c)[0]
            if len(non_zero):
                first[s] = non_zero[0]
                last[s] = non_zero[-1] + 1

        for ship in range(xy_data.shape[1]):
            times = (first[ship], last[ship])
            path = xy_data[first[ship]: last[ship], ship]

            ship_path = ShipPath(ID=ship, positions=[])
            if ship == 0:  # Ship 0 is always the harbor
                ship_path.positions.append(Position(time=m, pos=np.array([0, 0]), ship=ship_path))

            else:  # Every other ship besides the harbor
                for t, p in zip(list(range(times[0], times[-1] + 1)), path):
                    ship_path.positions.append(Position(time=t, pos=p, ship=ship_path))
            data[ship] = ship_path

        return data

    def visualize_path(self, path, seq, times):
        """
        Plot the paths of the ships and the path of the service ship in the working area
        :param path: Takes a list of Positions
        :return:
        """

        N = len(path)  # Number of labels

        # setup the plot
        fig, ax = plt.subplots(1, 1)

        # define the colormap
        cmap = plt.cm.jet
        # extract all colors from the .jet map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        # Bounds for the tick marks on color map
        bounds = np.linspace(0, N, N + 1)

        # Path positions for service ship
        x_path = []
        y_path = []
        x_first = []
        y_first = []
        x = []
        y = []
        tags = []  # Tags used for colors

        # Add service ship contact position for each shipZ
        for i, pos in enumerate(path):
            print(i)
            first = self.paths[seq[i]].positions[0]
            x_first.append(first[0])
            y_first.append(first[1])

            # Append paths for each of the individual ships
            for p in self.paths[seq[i]].positions:
                x.append(p.pos[0])
                y.append(p.pos[1])
                tags.append(N)
            N -= 1

            x_path.append(pos[0])
            y_path.append(pos[1])
            plt.annotate(str(times[i]), xy=(pos[0], pos[1]), fontsize='large')

        # The paths of the ships with coloring
        full_paths = ax.scatter(x, y, c=tags, cmap=cmap, alpha=0.5)
        plt.scatter(x_first, y_first, c='k', alpha=1, s=5)
        # The path of the service ship with marking
        plt.plot(x_path, y_path, c='red', ls='-', lw=2, marker='*', ms=10)

        # create the colorbar
        cb = plt.colorbar(full_paths, spacing='proportional', ticks=bounds)
        cb.set_label('Ships')
        ax.set_title('Service Ship Path')
        plt.show()


    def match_ampl(self, seq, t_seq):
        x = [0]
        y = [0]

        for s, t in zip(seq, t_seq):
            positions = self.paths[s]
            for p in positions:
                if p.time == t:
                    x.append(p[0])
                    y.append(p[1])
        x.append(0)
        y.append(0)

        d = 0
        d2 = []
        path = np.column_stack([x, y])
        for p in range(len(path) - 1):
            d += distance(path[p + 1], path[p])
            d2.append(distance(path[p + 1], path[p]))
        return path, d


def solve_dp(seq, data):
    ships = [(np.zeros((1)), np.zeros((1, 2)))]
    for s in seq:
        c = np.all(xy_data[:, s] != 0, axis=1)
        nz = np.where(c)[0]

        ships.append((nz, data[nz, s]))

    ships.append((np.array([1e16]), np.zeros((1, 2))))

    P = np.zeros((1, 2))
    T = [0]
    path = [[[0, 0]]]
    sched = [[0]]

    for k in range(len(ships)-1):

        time2, points2 = ships[k+1]

        D = cdist(points2, P)
        X = np.argsort(D, axis=1)

        _p = []
        _t = []
        _path = []
        _sched = []

        for j in range(X.shape[0]):

            for i in X[j]:

                next_time = T[i] + travel_time(D[j, i]) + 0.6

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

def distance(n1, n2):
    return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)

if __name__ == "__main__":
    # seq = [56, 26, 33, 8, 12]
    seq = [8, 5, 30, 63, 4]

    # Data
    x_data = np.genfromtxt("data/x.csv", delimiter=",")
    y_data = np.genfromtxt("data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    # number of ships in the working area in time [0, T]
    n = xy_data.shape[1]

    # Number of time slots in time [0, T]
    m = xy_data.shape[0]

    first = np.full(n, m, dtype=np.int)  # First slot when ship i is in the work area
    last = np.full(n, 0, dtype=np.int)  # Last slot when ship i is in the work area

    # For loop to fill out first, and last array
    for s in range(n):
        c = np.all(xy_data[:, s] != 0, axis=1)
        non_zero = np.where(c)[0]
        if len(non_zero):
            first[s] = non_zero[0]
            last[s] = non_zero[-1] + 1


    path, times = solve_dp(seq, xy_data)
    path = path[0]

    prob = Problem(xy_data)
    times[0][-1] = 0
    prob.visualize_path(path[1:-1], seq + [0], times[0][1:])
    d = 0
    for p in range(len(path)-1):
        d += distance(path[p+1], path[p])
    print(f"TOTAL DISTANCE: {d}")


    # path_ampl, dist = prob.match_ampl(seq=[0, 56, 26, 33, 8, 12, 0], t_seq=[0, 12, 13, 18, 23, 50, 50])
    path_ampl, dist = prob.match_ampl(seq=[0, 8, 5, 30, 63, 4, 0], t_seq=[0, 9, 12, 24, 36, 51, 71])
    prob.visualize_path(path_ampl[1:-1], [8, 5, 30, 63, 4, 0], [9, 12, 24, 36, 51, 71])
    print(f"TOTAL AMPL DISTANCE: {dist}")


