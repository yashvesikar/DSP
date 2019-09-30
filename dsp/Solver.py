import math
from typing import Type

import numpy as np
from scipy.spatial.distance import cdist

from dsp.Problem import Problem
from dsp.Visualize import Visualizer


class State:

    def __init__(self, shape, value):
        self.indexes = np.full(shape, value, dtype=np.int)
        self.schedule = np.full(shape, value, dtype=np.int)
        self.distances = np.full(shape, value, dtype=np.float)
        # self.positions = np.full(shape, value, dtype=np.float)
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
    def distance(n1, n2):
        return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)

    @staticmethod
    def travel_time(d):
        v = 46.3  # in km/h
        t = d / v

        w = 5 / 60

        return np.floor(t / (w + 0.1))

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

    def check_feasible(self, seq, times):
        PROC_TIME = 0.6
        time_windows = []
        travel_times = []
        for s in range(len(seq) - 1):
            s0 = self.problem.get_ship(seq[s])
            s1 = self.problem.get_ship(seq[s + 1])

            d = self.distance(s0.get_position(times[s]), s1.get_position(times[s + 1]))
            travel_times.append(self.travel_time(d) + PROC_TIME)
            time_windows.append(times[s + 1] - times[s])

        print(f"TRAVEL WINDOWS: {time_windows}")
        print(f"TRAVEL TIMES: {travel_times}")

    def construct_from_states(self, states, return_path=False, return_distance=False):

        def construct_path_from_schedule(sched):
            path = []
            for t, s in zip(sched, self.seq):
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

    def solve_sequence(self):
        """

        :param seq:
        :return:
        """
        assert len(self.seq) > 0

        seq = self.seq[:]

        ships = [self.problem.get_ship(s) for s in seq]

        Positions = np.zeros((1, 2))
        Times = [0]
        self.states.append(State(shape=ships[0].get_times(array=True).shape, value=0))

        for k in range(len(ships) - 1):

            s2 = ships[k + 1]

            time2, points2 = s2.get_times(array=True), s2.get_positions()

            # Returning to the harbor
            if ships[k + 1].id == 0:
                time2, points2 = np.array([time2[-1]]), np.array([points2[-1]])
            distance_matrix = np.atleast_2d(cdist(points2, Positions))
            total_distance = np.copy(distance_matrix)

            last_state = self.states[-1]
            for l, dist in enumerate(last_state.distances):
                if np.isnan(dist) or not np.all(dist):
                    break
                total_distance[:, l] += dist

            _positions = []
            _times = []
            _state = State(shape=time2.shape, value=np.nan)

            decision_matrix = np.argsort(total_distance, axis=1)

            for j in range(decision_matrix.shape[0]):
                # For every time slot the next ship is available

                for i in decision_matrix[j]:
                    # Based on the feasible paths until this point, check which paths can be expanded

                    travel = self.travel_time(distance_matrix[j, i])
                    next_time = Times[i] + travel + 0.6

                    if next_time <= time2[j]:
                        D = total_distance[j, i]
                        _positions.append(points2[j])
                        _times.append(time2[j])
                        _state.add_to_schedule(prev_ind=i, time=time2[j], dist=D)
                        break

            self.states.append(_state)
            Positions = _positions
            Times = _times

        Path, Schedule = self.construct_from_states(self.states, True)
        return Path, Schedule

    def next(self, s):

        # At first the sequence will be empty, the first node added will be the harbor
        if len(self.seq) <= 0:
            self.initalize()
            return
        last_ship = self.problem.get_ship(self.seq[-1])
        last_state = self.states[-1]
        if len(last_state) <= 0:
            # If this path is infeasible do not continue, cut the branch
            return False
        self.seq.append(s)
        current_ship = self.problem.get_ship(s)

        Times = last_state.schedule

        # Get the positions of the previous ship at the appropriate times
        Positions = last_ship.get_positions(range=(Times[0], Times[-1]))
        if last_ship.id == 0:
            # If the previous ship was the harbor need all the positions
            Positions = last_ship.get_positions()[np.arange(len(Times))]

        # if len(Positions) <= 0:
        #     # If there are no past positions that were feasible cut the branch
        #     return
        current_ship_time, current_ship_pos = current_ship.get_times(array=True), current_ship.get_positions()

        # Returning to the harbor
        if current_ship.id == 0:
            current_ship_time, current_ship_pos = np.array([current_ship_time[-1]]), np.array([current_ship_pos[-1]])
        distance_matrix = np.atleast_2d(cdist(current_ship_pos, Positions))
        total_distance = np.copy(distance_matrix)
        for l, dist in enumerate(last_state.distances):
            if np.isnan(dist):
                break
            total_distance[:, l] += dist
        dist_copy = np.copy(total_distance)

        _state = State(shape=current_ship_time.shape, value=np.nan)

        total_time_to_next = self.travel_time(distance_matrix) + 0.6 + Times

        # Where feasible solutions exist, keeps track of columns

        # eliminate all infeasible solutions
        total_time_to_next[total_time_to_next > current_ship_time[:, None]] = np.inf
        dist_copy[total_time_to_next > current_ship_time[:, None]] = np.inf
        # state_indexes = np.argmin(total_time_to_next, axis=1)
        state_indexes = np.argmin(dist_copy, axis=1)

        time_to_next_best = total_time_to_next[np.arange(len(total_time_to_next)), state_indexes]

        feasible = np.where(time_to_next_best != np.inf)[0]
        _state.indexes = state_indexes[feasible]
        _state.schedule = current_ship_time[feasible]
        _state.distances = total_distance[feasible, _state.indexes]

        # for j in range(decision_matrix.shape[0]):
        #     # For every time slot the next current_ship is available
        #
        #     for i in decision_matrix[j]:
        #         # Based on the feasible paths until this point, check which paths can be expanded
        #
        #         travel = self.travel_time(distance_matrix[j, i])
        #
        #
        #         next_time = Times[i] + travel + 0.6
        #
        #         # assert total_time_to_next[j, i] == next_time
        #
        #         # print(total_time_to_next[j, i], next_time)
        #         if next_time <= current_ship_time[j]:
        #             D = total_distance[j, i]
        #             _state.add_to_schedule(prev_ind=i, time=current_ship_time[j], dist=D)
        #             break

        self.states.append(_state)


if __name__ == "__main__":
    # Data
    x_data = np.genfromtxt("../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../data/y.csv", delimiter=",")

    xy_data = np.stack([x_data, y_data], axis=2)

    P = Problem(xy_data)

    seq = [0, 8, 5, 30, 63, 4, 0]
    # seq = [56, 26, 33, 8, 12, 0]
    # seq = [5, 4, 0]
    S = DPSolver(P, seq=[])
    # S.initalize()
    for i in seq:
        S.next(i)
    sched, dist = S.construct_from_states(S.states, return_distance=True)
    print(sched, dist)
    print(S.states[-1].distances[0])
    # print(S.states[-1].schedule)
    # pos, sched = S.solve_sequence()

    # d = 0
    # for p in range(len(pos[0]) - 1):
    #     d += S.distance(pos[0][p + 1], pos[0][p])
    # print(f"TOTAL DISTANCE: {d}")
    #
    # # Viz = Visualizer(P)
    # # Viz.visualize_path(seq, sched[0])
    #
    ampl = S.match_ampl(seq, sched)
    print(f"AMPL SOL: {ampl}")
    #
    # S.check_feasible(seq, sched)
