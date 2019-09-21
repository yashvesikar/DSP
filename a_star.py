#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import heapq
import matplotlib.pyplot as plt

import matplotlib as mpl
import math
import heapq

# Data
x_data = np.genfromtxt("data/x.csv", delimiter=",")
y_data = np.genfromtxt("data/y.csv", delimiter=",")

xy_data = np.stack([x_data, y_data], axis=2)


# Delete any column that has no ship in the working area at any time slot
# xy_data = np.delete(xy_data, np.unique(np.where(~xy_data.any(axis=0))[0]), axis=1)


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

class AStarNode:
    def __init__(self, position, parent=None, harbor=0):
        self.position = position
        self.current_path = []
        self.distance = 0
        self.harbor = harbor  # 1 if departing, 2 if returning
        self.f = 0
        self.g = 0
        self.h = 0
        self.parent = parent

    def __repr__(self):
        return f"Ship: {self.position.ship.id}, position: {self.get_coords()}, time: {self.get_time()}"

    def __str__(self):
        return f"Ship: {self.position.ship.id}, position: {self.get_coords()}, time: {self.get_time()}"

    def __lt__(self, other):
        """
        Less than comparator, needed for heapq
        :param other:
        :return:
        """
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

    def set_values(self, values):
        self.f, self.g, self.h = values

    def get_coords(self):
        return self.position.get_coords()

    def get_time(self):
        return self.position.get_time()

    def get_id(self):
        return self.position.get_id()

    def get_position(self):
        return self.position

    def get_time_range(self):
        t = self.position.ship.time
        return t[1] - t[0]


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

    def create_harbor(self):
        return AStarNode(self.paths[0].positions[0], harbor=1)

    def distance(self, n1, n2):
        return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)


    def visualize_path(self, path):
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
        x = []
        y = []
        tags = []  # Tags used for colors

        # Add service ship contact position for each shipZ
        for i, pos in enumerate(path):

            # Append paths for each of the individual ships
            for p in pos.ship.positions:
                x.append(p.pos[0])
                y.append(p.pos[1])
                tags.append(N)
            N -= 1

            x_path.append(pos[0])
            y_path.append(pos[1])
            plt.annotate(str(pos.ship.id), xy=(pos[0], pos[1]), fontsize='large')

        # The paths of the ships with coloring
        full_paths = ax.scatter(x, y, c=tags, cmap=cmap, alpha=0.5)

        # The path of the service ship with marking
        plt.plot(x_path, y_path, c='red', ls='-', lw=2, marker='*', ms=10)

        # create the colorbar
        cb = plt.colorbar(full_paths, spacing='proportional', ticks=bounds)
        cb.set_label('Ships')
        ax.set_title('Service Ship Path')
        plt.show()


    def solve_greedy(self, seq):
        """
        Implents the A* graph search algortithm to find a lower bound path for the problem
        Note:
        This is the greedy method of the problem where distance to the harbor is always minimized
        - Does not utilize time windows
        - Used a queue of size 1
        - Does not account for travel time - Travel is instantaneous
        Heuristic is the distance to the harbor. Therefore points closer to the harbor are preferred.
        :param seq: Sequence of ships to visit
        :return: List of positions of the path of the service ship, and the total distance covered
        """
        # Starting position is the harbor, create an A* node for the harbor
        HARBOR = self.create_harbor()
        HARBOR_POS = HARBOR.get_coords()

        # put the start node on the open list and leave its f value at 0
        op = [HARBOR]
        heapq.heapify(op)

        # Visited positions
        closed = set()

        while op:
            # Get the current A* node and add it to the closed/visited set
            current = heapq.heappop(op)
            # Empty out the op queue every time because we are taking the greedy solution
            # op = []
            # heapq.heapify(op)
            closed.add(current)

            # If the current node is the harbor and you are returning
            if current.harbor and current.harbor == 2:
                path = []
                distance = 0
                c = current

                if c is HARBOR:
                    c = c.parent
                    HARBOR.parent = None

                p = c.parent

                while c and p:
                    path.append(c.get_position())
                    distance += c.distance
                    c = p
                    p = c.parent
                return path[::-1], distance

            # If the current node is not the harbor
            else:
                # If the current node is the harbor and you are departing
                if current.harbor and current.harbor == 1:
                    current.harbor = 2

                # Seek the next ship in the sequence if it exists
                if seq:

                    positions = self.paths[seq.pop(0)].positions

                    for pos in positions:

                        child = AStarNode(position=pos, parent=current)
                        if child in closed:
                            continue
                        else:


                            child_pos = child.get_coords()

                            child.g = current.g + 1
                            # child.h = self.distance(child_pos, HARBOR_POS)
                            child.h = self.distance(child_pos, current.get_coords())
                            child.f = child.g + child.h

                            # Don't quite understand this condition
                            # for op_node in op:
                            #     if child == op_node and child.g > op_node.g:
                            #         continue
                            # if child in op and child.g > op[0].g:
                            #     # If the child is in the open list already and the g value is worse than min g in open list
                            #     continue
                            # if child in op and child.g >
                            # else:
                            # update the a* node stats
                            child.current_path.append(current)
                            child.distance += self.distance(child_pos, current.get_coords())

                            heapq.heappush(op, child)

                # If the sequence is empty then move back to the harbor
                else:
                    HARBOR.parent = current
                    heapq.heappush(op, HARBOR)


    def solve_relaxed(self, seq):
        """
        Implements the A* graph search algorithm to find a lower bound path for the problem
        Note:
        This is the greedy method of the problem where distance to the harbor is always minimized
        - Utilized relaxed time windows, time to visit is += the number of time slots
            it takes the service ship to cross the entire work area
        - Used a queue of size 1
        - Travel time is accounted for. Children are not added to the queue unless they can be visited in time
        Heuristic is the distance to the harbor. Therefore points closer to the harbor are preferred.
        :param seq: Sequence of ships to visit
        :return: List of positions of the path of the service ship, and the total distance covered
        """
        # Initialize
        self.time = 0  # reset time
        # Starting position is the harbor, create an A* node for the harbor
        HARBOR = self.create_harbor()
        HARBOR_POS = HARBOR.get_coords()

        # put the start node on the open list and leave its f value at 0
        op = [HARBOR]
        heapq.heapify(op)

        # Visited positions
        closed = set()

        while op:
            # Get the current A* node and add it to the closed/visited set
            current = heapq.heappop(op)
            if not current.harbor:
                self.time = current.get_time()
            # Empty out the op queue every time because we are taking the greedy solution
            # op = []
            op = op[:10]
            heapq.heapify(op)
            closed.add(current)

            # If the current node is the harbor and you are returning
            if current.harbor and current.harbor == 2:
                path = []
                distance = 0
                c = current

                if c is HARBOR:
                    c = c.parent
                    HARBOR.parent = None

                p = c.parent

                while c and p:
                    path.append(c.get_position())
                    distance += c.distance
                    c = p
                    p = c.parent
                return path[::-1], distance

            # If the current node is not the harbor
            else:
                # If the current node is the harbor and you are departing
                if current.harbor and current.harbor == 1:
                    current.harbor = 2

                # Seek the next ship in the sequence if it exists
                if seq:

                    positions = self.paths[seq.pop(0)].positions

                    for pos in positions:

                        # TODO: Timing checks go here

                        # If the time of the position is less than the current time then continue
                        if pos.time <= self.time:  # + time to travel to that point
                            continue

                        child = AStarNode(position=pos, parent=current)
                        if child in closed:
                            continue
                        else:


                            child_pos = child.get_coords()

                            child.g = current.g + 1
                            child.h = self.distance(child_pos, HARBOR_POS) - (child.get_time() - child.position.ship.positions[0].time)
                            child.f = child.g + child.h

                            # Don't quite understand this condition
                            # for op_node in op:
                            #     if child == op_node and child.g > op_node.g:
                            #         continue
                            # if child in op and child.g > op[0].g:
                            #     # If the child is in the open list already and the g value is worse than min g in open list
                            #     continue
                            # if child in op and child.g >
                            # else:
                            # update the a* node stats
                            child.current_path.append(current)
                            child.distance += self.distance(child_pos, current.get_coords())

                            heapq.heappush(op, child)

                # If the sequence is empty then move back to the harbor
                else:
                    HARBOR.parent = current
                    heapq.heappush(op, HARBOR)

if __name__ == "__main__":
    seq = [56, 26, 33, 8, 12]
    og_seq = seq[:]
    prob = Problem(xy_data)
    # path, distance = prob.solve_relaxed(seq=seq)
    path, distance = prob.solve_greedy(seq=seq)

    prob.visualize_path(path)
    for p in path:
        print(p)
    print(f"DISTANCE MIN NEXT: {distance}")
