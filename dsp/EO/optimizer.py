import itertools

import numpy as np
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize

from dsp.Problem import Problem as ShipProblem
from dsp.Runner import SequenceSolver
from dsp.Solver import DPSolver, solve_sequence
from dsp.Window import sliding_window


class FixedNumberOfShipsProblem(Problem):
    def __init__(self, problem, n_ships):
        # get ships in working area and do not consider the harbor
        self.ships = set(problem.ships_in_working_area()[1:])
        self.n_avail = len(self.ships)

        # initialize the problem
        super().__init__(n_var=n_ships, n_obj=1, n_constr=1, elementwise_evaluation=True)

        # save the original problem
        self.data = problem

    def _evaluate(self, x, out, *args, algorithm=None, **kwargs):
        seq = [0] + x.tolist() + [0]
        solver = solve_sequence(self.data, seq)
        out["F"] = solver.dist
        out["G"] = 0.0 if solver.feasible else 1.0
        out["solver"] = solver


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.int)
        for k in range(n_samples):
            X[k] = (np.random.permutation(problem.n_avail) + 1)[:problem.n_var]
        return X


class LevelOrderSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        level_order_solver = SequenceSolver(problem=problem.data, height_limit=problem.n_var)
        truncation_args = {'limit': 100, 'method': "distance"}

        result = level_order_solver.sequence_search(available=problem.ships, truncation_args=truncation_args)

        X = np.array([e[1:-1] for e in result['all_seq'][-1]])
        X = X[np.random.permutation(len(X))[:n_samples]]

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, algorithm=None, **kwargs):
        solvers = algorithm.pop.get("solver")
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), -1)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            i = np.random.randint(problem.n_var)

            _X[0, k, :i] = p1[:i]
            _X[0, k, i:] = p2[i:]

            _X[1, k, :i] = p2[:i]
            _X[1, k, i:] = p1[i:]

            _X[0, k] = p1
            _X[1, k] = p2

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, algorithm=None, **kwargs):
        solvers = algorithm.pop.get("solver")

        for k in range(X.shape[0]):
            x = X[k]
            # solver = solvers[k, 0]
            # sched = solver.schedule[1:-1]

            i = np.random.randint(problem.n_var)

            _start, _end = 0, int(problem.data.m)
            if i - 1 >= 0:
                _start = problem.data.get_ship(x[i - 1]).times[0]
            if i + 1 <= problem.n_var - 1:
                _end = problem.data.get_ship(x[i + 1]).times[0]

            avail = set(problem.data.ships_in_working_area(start=_start, end=_end)) - set(x) - set([0])

            if len(avail) > 0:
                x[i] = np.random.choice(list(avail))

        return X


def load():
    x_data = np.genfromtxt("../../data/x.csv", delimiter=",")
    y_data = np.genfromtxt("../../data/y.csv", delimiter=",")
    xy_data = np.stack([x_data, y_data], axis=2)
    problem = ShipProblem(xy_data)
    return problem



def my_sliding_window(k, n, S, return_first=False):
    success = 0
    fail = 0
    seq, sched, problem = S.seq, S.schedule, S.problem
    best_dist = 1e10
    best_solver = None
    n += 1
    for i in range(len(seq)-n):
        avail = problem.ships_in_working_area(start=sched[i], end=sched[i + n])
        avail = set(avail) - set(seq[:i+1] + seq[i+n:])
        if 0 in avail:
            avail.remove(0)

        sub_seq = list(itertools.permutations(avail, k))

        # MOD
        sub_seq = [s for s in sub_seq if len(set(seq[i+1:i+n]) - set(s)) == 0]

        for ship in sub_seq:
            _seq = seq[:i+1] + list(ship) + seq[i+n:]
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




def transition(pop):
    n_individuals, n_var = pop.get("X").shape

    X = np.zeros((n_individuals, n_var+1))

    for k, ind in enumerate(pop):
        solver = ind.data["solver"][0]
        X[k] = my_sliding_window(2, 1, solver, return_first=True).seq[1:-1]

    return X


def solve(seq_length, sampling=None):

    if sampling is None:
        sampling = LevelOrderSampling()

    subproblem = FixedNumberOfShipsProblem(problem, seq_length)

    algorithm = GA(
        pop_size=20,
        sampling=sampling,
        crossover=BinaryCrossover(),
        mutation=MyMutation(),
        eliminate_duplicates=True)

    res = minimize(subproblem,
                   algorithm,
                   ('n_gen', 5),
                   seed=1,
                   verbose=False)

    return res.X, res.F[0], res.pop


if __name__ == "__main__":
    problem = load()

    ret = []
    X, F, pop = solve(1)
    n_ships = 2

    while X is not None:
        S = transition(pop)
        X, F, pop = solve(n_ships, sampling=S)
        print(n_ships, F)

        n_ships += 1

