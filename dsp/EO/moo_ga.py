import copy
from math import ceil

import numpy as np

from dsp.EO.level_ga import load
from dsp.Runner import SequenceSolver
from dsp.Solver import solve_sequence
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize


class MOOProblem(Problem):
    def __init__(self, problem):
        self.ships = set(problem.ships_in_working_area()[1:])
        self.n_avail = len(self.ships)
        super().__init__(n_var=1, n_obj=1, n_constr=1, elementwise_evaluation=True)
        self.data = problem

    def _evaluate(self, x, out, *args, **kwargs):
        seq = [0] + x[0] + [0]
        solver = solve_sequence(self.data, seq)
        out["F"] = solver.dist
        out["G"] = 0.0 if solver.feasible else 1.0
        out["solver"] = solver


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        level_order_solver = SequenceSolver(problem=problem.data, height_limit=1000)
        truncation_args = {'limit': 25, 'method': "distance"}

        ret = level_order_solver.sequence_search(available=problem.ships,
                                                 truncation_args=truncation_args, verbose=False)["selected"]

        n_each_seq = ceil(n_samples / len(ret))

        X = []
        for level in ret:
            seq = [e[-1][1:] for e in level[:n_each_seq]]
            X.extend(seq)

        return np.array(X, dtype=np.object)[:, None]


def crossover(p_a, p_b, s_a):
    I = np.random.permutation(len(p_a))
    for i in I:
        _start = max(i - 3, 0)
        _end = min(i + 3, len(p_a))
        J = np.array(range(_start, _end))
        J = J[np.random.permutation(len(J))]

        for j in J:
            prefix, suffix = p_a[:i], p_b[j:]
            suffix = [s for s in suffix if s not in prefix]

            if len(suffix) > 0:

                solver = copy.copy(s_a)
                solver.feasible = False
                solver.states = s_a.states[:i + 1]
                solver.seq = s_a.seq[:i + 1]

                solver.next(suffix[0])
                solver.feasible = len(solver.states[-1].times) > 0

                if solver.feasible:
                    return prefix + suffix

    return p_a


class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def do(self, problem, pop, parents, algorithm=None, **kwargs):
        solvers = algorithm.pop.get("solver")

        X = []
        for k, (a, b) in enumerate(parents):
            s_a, s_b = solvers[a, 0], solvers[b, 0]
            p_a, p_b = pop[a].X[0], pop[b].X[0]

            X.append([crossover(p_a, p_b, s_a)])
            X.append([crossover(p_b, p_a, s_b)])

        off = pop.new("X", X)
        return off


class MyMutation(Mutation):
    def do(self, problem, pop, **kwargs):
        return pop.new("X", pop.get("X"))


# the input is the current population and a list of other populations.
# the function returns if an individual in pop is equal to any other individual
# in any other population.
def func_is_duplicate(pop, *other, **kwargs):
    if len(other) == 0:
        return np.full(len(pop), False)

    # value to finally return
    is_duplicate = np.full(len(pop), False)

    H = set()
    for e in other:
        for val in e:
            s = str(val.X[0])
            H.add(s)

    for i, (val,) in enumerate(pop.get("X")):
        s = str(val)
        if s in H:
            is_duplicate[i] = True
        H.add(s)

    return is_duplicate


if __name__ == "__main__":
    problem = load()

    np.random.seed(1)

    my_problem = MOOProblem(problem)

    algorithm = GA(
        pop_size=100,
        sampling=MySampling(),
        crossover=MyCrossover(),
        mutation=MyMutation(),
        eliminate_duplicates=func_is_duplicate)

    res = minimize(my_problem,
                   algorithm,
                   ('n_gen', 100),
                   seed=1,
                   verbose=False)

    print(res.F)