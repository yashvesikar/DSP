import copy
import math
import numpy as np
from dsp.Problem import Problem, load_problem
from collections import namedtuple
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

class Result:
    def __init__(self):
        self.feasible = False
        self.seq = None
        self.schedule = None
        self.distance = None

    def __copy__(self):
        obj = Result()
        obj.feasible = self.feasible
        obj.seq = self.seq[:]
        obj.schedule = self.schedule[:] if self.schedule is not None else None
        obj.distance = self.distance
        return obj

class DPSolver:
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.states = kwargs.get('states') if kwargs.get('states') is not None else []
        self.seq = kwargs.get('seq')
        self.result = None

    def __repr__(self):
        return f"Seq: {self.seq}"

    def __copy__(self):
        obj = DPSolver(self.problem)
        obj.seq = self.seq[:]
        obj.states = self.states[:]
        obj.result = Result()
        return obj

    def clear(self):
        self.states = []
        self.seq = []
        self.schedule = []
        self.feasible = False
        self.solution = None
        self.dist = 0

    def pop_state(self):
        self.states.pop()
        self.seq.pop()

    def last_state(self):
        if self.states:
            return self.states[-1]
        return None

    @staticmethod
    def squared_dist(A, B):
        return np.sqrt(((A[:, None] - B[None, :]) ** 2).sum(axis=2))

    @staticmethod
    def distance(n1, n2):
        return math.sqrt((n1[0] - n2[0]) ** 2 + (n1[1] - n2[1]) ** 2)

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

    def simulate_path(self, seq, times):
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

    def update_final_state(self, feasible=True):
        if feasible:
            self.feasible = True
            self.schedule, self.dist = self.get_result(self.states, return_distance=True)
        else:
            self.feasible = False
            self.schedule, self.dist = None, None


    def get_result(self, **kwargs):
        """

        :param kwargs:
        :return:
        """
        # if self.result and self.result.seq == self.seq:
        #     return self.result

        result = Result()
        seq = self.seq
        states = self.states

        # Infeasible solution
        if len(states) != len(seq) or len(states[-1]) == 0 or seq[0] != 0 or seq[-1] != 0:
            schedule = None
            path = None
            result.distance = 1e10
            result.feasible = False
        else:
            prev_index = -1
            schedule = []
            for state in states[::-1]:
                schedule.append(state.schedule[prev_index])
                prev_index = int(state.indexes[prev_index])

            schedule = schedule[::-1]
            path = []
            for t, s in zip(schedule, seq):
                ship = self.problem.get_ship(s)
                path.append(ship.get_position(int(t)))

            if len(schedule) == len(seq) and seq[0] == 0 and seq[-1] == 0:
                result.feasible = True

            result.distance = self.path_distance(path)

        result.seq = seq
        result.schedule = schedule
        result.path = path
        self.result = result
        return result

    def initalize(self):
        """

        :return:
        """
        self.seq.append(0)
        harbor = self.problem.get_ship(0)
        harbor_times = harbor.get_times(array=True)
        self.states.append(State(shape=harbor_times.shape, value=0))

    def next(self, s, modify_seq=True):

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
        if modify_seq:
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

        if current_ship.id == 0:
            # When returning to the harbor we want to select only the single best time to return
            # The time is selected based on minimum distance to the harbor
            # Since harbor is always at the same position we can use this to simplify the decision.
            # Previously Always returned at Time 72, now will return at earliest & best feasible time

            # Index of minimum distance in total_distance matrix
            ind = np.unravel_index(np.argmin(total_distance, axis=None), total_distance.shape)

            # Simplified distance and time matrixes/arrays
            total_time_to_next = np.atleast_2d(total_time_to_next[ind[0], :])
            total_distance = np.atleast_2d(total_distance[ind[0], :])
            current_ship_time = np.array([current_ship_time[ind[0]]])


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
        if len(feasible) > 0:
            return True
        return False

def solve_sequence(problem, seq, **kwargs):
    solver = DPSolver(problem=problem, seq=[])

    assert len(seq) > 2 and seq[0] == 0 and seq[-1] == 0

    for s in seq:
        solver.next(s)

    result = solver.get_result()

    solver.result = result

    return solver

if __name__ == "__main__":
    P = load_problem(T=6)
    seq = [0, 32, 6, 4, 0]
    solver = solve_sequence(problem=P, seq=seq)