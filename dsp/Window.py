import itertools
import numpy as np
import copy
from dsp.Solver import solve_sequence
import heapq


def sliding_window(insert, remove, S, return_first=False):
    """

    :param insert: Number of ships to expand by
    :param remove: Number of ships to remove
    :param S: DPSolver instance
    :return:
    """
    success = 0
    fail = 0
    best_dist = 1e10
    best_solver = None
    # Algorithm works on gaps between ships not ships themselves, hence remove += 1
    remove += 1
    problem = load_problem()

    if isinstance(S, list):
        solver = solve_sequence(problem=problem, seq=S)
    elif isinstance(S, DPSolver):
        solver = copy.copy(S)
    else:
        print("FAIL SLIDING WINDOW")
        return
    sequence, schedule, distances = solver.seq[:], solver.schedule[:], [min(s.distances) for s in solver.states]

    for i in range(len(sequence) - remove):
        window = sequence[i+1:i + remove]

        start = min(S.states[i].schedule)  # Minimum feasible available time from the ship states before the window
        end = problem.get_ship(window[-1]).times[1]  # Maximum available time for the ship in WA following the window

        avail = problem.ships_in_working_area(start=start, end=end)
        avail = set(avail) - set(sequence[:i+1] + sequence[i + remove:])
        if 0 in avail:
            avail.remove(0)

        sub_seq = list(itertools.permutations(avail, insert))
        candidates = []  # Candidate solutions for full evaluation
        for ship in sub_seq:
            # Update the solver to the correct transition point
            while len(solver.seq) != len(sequence[:i+1]):
                solver.pop_state()

            # Evaluate the transition with the new permutation
            skip = False
            for l, j in enumerate(ship):
                b = solver.next(j)
                new_sched = solver.states[-1].schedule
                new_dist = min(solver.states[-1].distances) if len(solver.states[-1].distances) else 1e6
                # If the new schedule has less possible transitions or the new minimum distance is greate
                if len(new_sched) < len(schedule) or new_dist > distances[i+1+l]:
                    skip = True
                    break
            if skip:  # Move to next permutation
                fail += 1
                continue
            else:  # Update candidate permutations based on distance
                success += 1
                heapq.heappush(candidates, (min(solver.states[-1].distances), copy.copy(solver)))

        # Evaluate all candidate solutions
        for c in range(len(candidates)):
            d, solver = heapq.heappop(candidates)
            for s in sequence[i + remove:]:
                solver.next(s)

            res = solver.get_result(solver.states, return_path=False, return_distance=True)
            if res is not None:
                if return_first:
                    return solver

                if res[-1] < best_dist:
                    best_dist = res[-1]
                    best_solver = solver
    return best_solver


def old_sliding_window(k, n, S, return_first=False):
    success = 0
    fail = 0
    seq, sched, problem = S.seq, S.schedule, S.problem
    best_dist = 1e10
    best_solver = None
    n += 1
    for i in range(len(seq) - n):
        avail = problem.ships_in_working_area(start=sched[i], end=sched[i + n])
        avail = set(avail) - set(seq[:i + 1] + seq[i + n:])
        if 0 in avail:
            avail.remove(0)

        sub_seq = list(itertools.permutations(avail, k))

        # # MOD
        # sub_seq = [s for s in sub_seq if len(set(seq[i + 1:i + n]) - set(s)) == 0]

        for ship in sub_seq:
            _seq = seq[:i + 1] + list(ship) + seq[i + n:]
            result = solve_sequence(problem=problem, seq=_seq)
            if result.feasible:
                success += 1

                # Return the first feasible solution
                if return_first:
                    return result

                if result.dist < best_dist:
                    best_dist = result.dist
                    best_solver = result

            else:
                fail += 1

    return best_solver


def ping_pong_solver(solvers, fast=False):

    length = 0
    solver = solvers[length]
    optimization_counts = [0] * len(solvers)
    terminate = False
    improved = False
    mode = 'optimize'  #Available modes: 'grow', 'shrink', 'optimize'
    while not terminate:
        if mode == 'optimize':
            intermediate_results = []
            for i in range(3):
                i += 1
                result = sliding_window(insert=i, remove=i, S=solver, return_first=fast)
                if result.seq != solver.seq and result.dist < solver.dist:
                    intermediate_results.append(result)
                    print(f"Optimized: {solver.seq} -> {result.seq}")
                    improved = True
            # No improvement in the sequence
            if not improved:
                # If the level above me has already been optimized once and the level below me has been optimized once
                if 0 < length < len(solvers) and optimization_counts[length + 1] > 1 and optimization_counts[length - 1]:
                    terminate = True
                    continue
                if length == len(solvers) and length > 0 and optimization_counts[length - 1]:
                    mode = 'shrink'
                    length -= 1

                else:
                    mode = 'grow'
                    length += 1
                solver = solvers[length]

            # Improvement in the sequence
            else:
                # Sort the improvements and take the best one
                intermediate_results = sorted(intermediate_results, key=lambda x: x.dist)
                solvers[length] = intermediate_results[0]
                solver = solvers[length]




if __name__ == "__main__":
    from dsp.Problem import Problem, load_problem
    from dsp.Solver import DPSolver
    from dsp.Runner import SequenceSolver
    import time
    import pprint as pp

    # Create sequence solver object
    P = load_problem()
    truncation_args = {'limit': 100, 'method': "distance"}
    # ALPHA = set(P.ships_in_working_area())
    # SeqSolver = SequenceSolver(problem=P, height_limit=5)
    # result = SeqSolver.sequence_search(available=ALPHA, truncation_args=truncation_args)
    sol = solve_sequence(problem=P, seq=[0, 32, 30, 63, 12, 0])
    # sol2 = solve_sequence(problem=P, seq=[0, 32, 30, 63, 12, 0])
    sol2 = solve_sequence(problem=P, seq=[0, 8, 5, 30, 63, 12, 0])

    # start = time.time()
    # result = old_sliding_window(1, 1, sol)
    # end = time.time()
    # print(f"Old Result 2 Time: {end-start}")
    # print(f"Old Result: {result}")
    #
    # start = time.time()
    # result2 = sliding_window(1, 1, sol2)
    # end = time.time()
    # print(f"Result Time:       {end-start}")
    # print(f"Result:     {result2}")
    ping_pong_solver([sol, sol2], fast=False)