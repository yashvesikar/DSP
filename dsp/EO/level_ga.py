import itertools

import numpy as np

from dsp.Problem import Problem as ShipProblem, load_problem

from dsp.Solver import solve_sequence
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.model.termination import SingleObjectiveToleranceBasedTermination
from pymoo.optimize import minimize

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
        out["F"] = solver.result.distance
        out["G"] = 0.0 if solver.result.feasible else 1.0
        out["solver"] = solver


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=np.int)
        for k in range(n_samples):
            X[k] = (np.random.permutation(problem.n_avail) + 1)[:problem.n_var]
        return X


class LevelOrderSampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        from dsp.HTSolver import HeuristicTreeSolver
        truncation_args = {'limit': 100, 'method': "distance"}
        level_order_solver = HeuristicTreeSolver(problem=problem.data, max_depth=problem.n_var, truncation_args=truncation_args)


        result = level_order_solver.solve()

        X = np.array([e.seq[1:-1] for e in result['selected'][-1]])
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

        for k in range(X.shape[0]):
            x = X[k]

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



class MyTermination(SingleObjectiveToleranceBasedTermination):

    def _decide(self):
        # now check the F space
        current = self.history[0][1].mean()
        last = self.history[-1][1].mean()

        # the absolute difference of current to last f
        f_tol_abs = last - current < self.f_tol_abs

        return not f_tol_abs


def transition(pop):
    X = []
    for k, ind in enumerate(pop):
        solver = ind.data["solver"][0]
        res = sliding_window(2, 1, solver, return_first=False, restrict_available=True)
        if res is not None:
            X.append(res.seq[1:-1])
    if len(X) <= 0:
        return None

    return np.row_stack(X)


def transition_simple(problem, pop):
    X = []
    for k, ind in enumerate(pop):
        x = ind.X.tolist()

        avail = set(problem.ships_in_working_area()) - set(ind.X) - set([0])
        ship = np.random.choice(list(avail))

        pos = np.random.randint(len(x))
        x.insert(pos, ship)

        X.append(x)


    return np.row_stack(X)



def solve(seq_length, problem, sampling=None):
    if sampling is None:
        sampling = LevelOrderSampling()

    subproblem = FixedNumberOfShipsProblem(problem, seq_length)
    termination = MyTermination(f_tol_abs=0.1)

    algorithm = GA(
        pop_size=100,
        sampling=sampling,
        crossover=BinaryCrossover(),
        mutation=MyMutation(),
        eliminate_duplicates=True)

    res = minimize(subproblem,
                   algorithm,
                   termination,
                   # ('n_gen', 20),
                   seed=1,
                   verbose=False)

    return res

def solve_level_ga(problem, verbose=False):
    n_ships = 1
    results = []

    while True:

        if n_ships == 1:
            res = solve(1, problem)
        else:
            S = transition(pop)
            if S is not None:
                res = solve(n_ships, problem, sampling=S)
            else:
                break

        results.append(res.opt.data["solver"][0])
        _X, _F, pop = res.X, res.F[0], res.pop

        # prepare for the next iteration
        I, X, F = pop.get("feasible", "X", "F")

        if verbose:
            print(n_ships, _F, _X.tolist(), res.algorithm.n_gen, len(I))

        if len(I) == 0:
            break


        pop = pop[I[:, 0]]
        n_ships += 1

    return {"solvers": results}

if __name__ == "__main__":
    np.random.seed(1)
    res = solve_level_ga(load_problem(6), verbose=True)

    print(res['solvers'])