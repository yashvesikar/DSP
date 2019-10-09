import math
import numpy as np
from dsp.Problem import Problem
from dsp.Visualize import Visualizer

class State:

    def __init__(self, shape=None, value=None):
        self.indexes = np.full(shape, value, dtype=np.int) if shape else None
        self.schedule = np.full(shape, value, dtype=np.int) if shape else None
        self.distances = np.full(shape, value, dtype=np.float) if shape else None
        self.times = np.full(shape, value, dtype=np.float) if shape else None
        self.i = 0

    def __repr__(self):
        return f"Sched: {self.schedule}"

    def __len__(self):
        return len(self.indexes)

    def add_to_schedule(self, prev_ind, time, dist):
        self.indexes[self.i] = prev_ind
        self.schedule[self.i] = time
        self.distances[self.i] = dist
        # self.positions[self.i] = pos
        self.i += 1


class DPSolver:
    def __init__(self, problem, seq, states=None):
        self.problem = problem
        self.states = states if states is not None else []
        self.seq = seq
        self.feasible = False


    def __repr__(self):
        return f"Seq: {self.seq}"

    def __copy__(self):
        return DPSolver(problem=self.problem, seq=self.seq[:], states=self.states[:])

    def pop_state(self):
        self.states.pop()
        self.seq.pop()

    def get_last_state(self):
        if self.states:
            return self.states[-1]
        return None

    @staticmethod
    def squared_dist(A, B):
        return np.sqrt(((A[:, None] - B[None, :]) ** 2).sum(axis=2))

    @staticmethod
    def distance(n1, n2):
        return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)

    @staticmethod
    def travel_time(d):
        v = 46.3  # in km/h
        t = d / v

        w = 5 / 60

        return t / w

    def get_most_recent_time_pos(self):
        """
        If the DPSolver has solved a feasible schedule we need the time slot and position of the last ship in the sequence

        This is used in order to make a distance calculation to determine which ships to expand to in the sequence search

        :return: time slot, position
        """
        assert self.feasible is True

        last_state = self.states[-1]
        last_ship_index = last_state.indexes[0]  # index of the optimal move from the last ship to harbor in the last ship state

        last_ship_id = self.seq[-2]
        last_ship = self.problem.get_ship(last_ship_id)

        last_ship_state = self.states[-2]
        last_ship_schedule = last_ship_state.schedule[last_ship_index]
        last_ship_pos = last_ship.get_position(last_ship_schedule)

        return last_ship_schedule, last_ship_pos, last_ship_id


    def path_distance(self, positions):

        d = 0
        for p in range(len(positions) - 1):
            d += self.distance(positions[p + 1], positions[p])

        return d

    def match_ampl(self, seq, times):
        x = []
        y = []

        for s, t in zip(seq, times):
            ship = self.problem.get_ship(s)
            pos = ship.get_position(t)
            x.append(pos[0])
            y.append(pos[1])

        d = 0
        d2 = []
        path = np.column_stack([x, y])
        for p in range(len(path) - 1):
            d += self.distance(path[p + 1], path[p])
            d2.append(self.distance(path[p + 1], path[p]))
        return d


    def construct_from_states(self, states, seq=None, return_path=False, return_distance=False):
        if seq is None:
            seq = self.seq

        if len(states) != len(seq) or len(states[-1]) == 0:
            # Infeasible solution
            return None

        def construct_path_from_schedule(sched):
            path = []
            for t, s in zip(sched, seq):
                ship = self.problem.get_ship(s)
                path.append(ship.get_position(int(t)))

            return path

        schedule = []
        result = []
        prev_index = -1
        for state in states[::-1]:
            schedule.append(state.schedule[prev_index])
            prev_index = int(state.indexes[prev_index])

        schedule = schedule[::-1]
        result.append(schedule)
        if return_path:
            path = construct_path_from_schedule(schedule)
            result.append(path)

        if return_distance:
            pos = construct_path_from_schedule(schedule)
            dist = self.path_distance(pos)
            result.append(dist)

        return result

    def initalize(self):
        self.seq.append(0)
        harbor = self.problem.get_ship(0)
        harbor_times = harbor.get_times(array=True)
        self.states.append(State(shape=harbor_times.shape, value=0))

    def solve(self, seq=None, return_path=False, return_distance=False):

        if seq is None:
            seq = self.seq

        assert len(seq) > 2 and seq[0] == 0 and seq[-1] == 0

        for s in seq:
            self.next(s)
        sol = self.construct_from_states(self.states, return_path=return_path, return_distance=return_distance)

        if sol is not None:
            self.feasible = True

        return sol

    def next(self, s):

        # At first the sequence will be empty, the first node added will be the harbor
        if len(self.states) <= 0 and s == 0:
            self.initalize()
            return True
        # Get last ship and last state
        last_ship = self.problem.get_ship(self.seq[-1])
        last_state = self.states[-1]
        if len(last_state) <= 0:
            # If this path is infeasible do not continue, cut the branch
            return False
        # Append the new ship onto the sequence
        self.seq.append(s)
        current_ship = self.problem.get_ship(s)

        # Get times and positions of the previous ship in the sequence
        Times = last_state.schedule

        Positions = last_ship.get_positions(range=(int(Times[0]), int(Times[-1])))
        if last_ship.id == 0:
            # If the previous ship was the harbor need all the positions
            Positions = last_ship.get_positions()[np.arange(len(Times))]

        # Get the times and positions of the current ship
        current_ship_time, current_ship_pos = current_ship.get_times(array=True), current_ship.get_positions()
        if current_ship.id == 0:
            # Returning to the harbor
            current_ship_time, current_ship_pos = np.array([current_ship_time[-1]]), np.array([current_ship_pos[-1]])

        # Calculate the distance between the 2 ship paths, and modify the total_distance matrix
        distance_matrix = np.atleast_2d(np.sqrt(((current_ship_pos[:, None] - Positions[None, :]) ** 2).sum(axis=2)))
        total_distance = np.copy(distance_matrix)
        total_distance += last_state.distances

        # New state object to store state of the current ship
        _state = State()

        # eliminate all infeasible solutions by setting the travel time to infinite
        total_time_to_next = (distance_matrix / 46.3) / (5 / 60)  # Store travel times to next ship

        if last_ship.id == 0:
            shape = total_time_to_next.shape
            # Newest to oldest
            total_time_to_next[total_time_to_next >= (current_ship_time[:, None] - Times)] = np.inf
            total_distance[total_time_to_next >= (current_ship_time[:, None] - Times)] = np.inf

        else:
            # newest to oldest
            total_time_to_next[total_time_to_next >= (current_ship_time[:, None] - Times + 0.4)] = np.inf
            total_distance[total_time_to_next >= (current_ship_time[:, None] - Times + 0.4)] = np.inf

        # Choose closest feasible position index for each of the current ship positions
        state_indexes = np.argmin(total_distance, axis=1)

        # Store the travel times to the closest feasible ships
        time_to_next_best = total_time_to_next[np.arange(len(total_time_to_next)), state_indexes]

        # Select feasible solutions indexes and update the state of the current ship path
        feasible = np.where(time_to_next_best != np.inf)[0]
        _state.indexes = state_indexes[feasible]
        _state.schedule = current_ship_time[feasible]
        _state.distances = total_distance[feasible, _state.indexes]
        _state.times = total_time_to_next[feasible, _state.indexes]
        self.states.append(_state)
        return True


if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)

    seq = [0, 8, 5, 30, 63, 4, 0]
    # seq = [0, 33, 8, 44, 32, 4, 0]
    # seq = [0, 56, 26, 33, 8, 12, 0]
    # seq = [0, 15, 5, 8, 44, 38, 4, 12, 23, 61, 28, 0]
    # seq = [0, 29, 32, 4, 0]
    S = DPSolver(P, seq=[])

    sched, dist = S.solve(seq=seq, return_distance=True)
    print(sched, dist)
    print(S.states[-1].distances[0])

    Viz = Visualizer(P)
    Viz.visualize_path(seq, sched)

    ampl = S.match_ampl(seq, sched)
    print(f"AMPL SOL: {ampl}")
