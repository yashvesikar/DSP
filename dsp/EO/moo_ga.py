import copy
import itertools
from math import ceil

import numpy as np

from dsp.EO.level_ga import load
from dsp.Runner import SequenceSolver
from dsp.Solver import solve_sequence
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.problem import Problem
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize


class MOOProblem(Problem):
    def __init__(self, problem):
        self.ships = set(problem.ships_in_working_area()[1:])
        self.n_avail = len(self.ships)
        super().__init__(n_var=1, n_obj=2, n_constr=1, elementwise_evaluation=True)
        self.data = problem

    def _evaluate(self, x, out, *args, **kwargs):
        val = x[0]
        if isinstance(val, str):
            # print(val)
            val = [int(e) for e in val[1:-1].split(",")]

        seq = [0] + val + [0]
        solver = solve_sequence(self.data, seq)
        out["F"] = np.array([- float(len(val)), solver.dist])
        out["G"] = 0.0 if solver.feasible else 1.0
        out["solver"] = solver

    def _calc_pareto_front(self, *args, **kwargs):
        ideal = [0.0, 0.0]
        nadir = [20.0, 210.0]
        return np.row_stack([ideal, nadir])


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        level_order_solver = SequenceSolver(problem=problem.data, height_limit=1000)
        truncation_args = {'limit': 25, 'method': "distance"}

        ret = level_order_solver.solve(available=problem.ships,
                                       truncation_args=truncation_args, verbose=False)["selected"]

        n_each_seq = ceil(n_samples / len(ret))

        X = []
        for level in ret:
            seq = [e[-1][1:-1] for e in level[:n_each_seq]]
            X.extend(seq)

        return [[str(row)] for row in X]


def crossover(p_a, p_b, s_a):
    I = np.random.permutation(len(p_a))
    for i in I:
        J = np.random.permutation(len(p_b))

        for j in J:
            prefix, suffix = p_a[:i], p_b[j:]
            suffix = [s for s in suffix if s not in prefix]
            # return prefix + suffix

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
            p_a, p_b = s_a.seq[1:-1], s_b.seq[1:-1]

            X.append(crossover(p_a, p_b, s_a))
            X.append(crossover(p_b, p_a, s_b))

        X = [[str(row)] for row in X]

        off = pop.new("X", X)
        return off

class MOOMutation(Mutation):

    def do(self, problem, pop, **kwargs):
        X = pop.get("X")

        for k, off in enumerate(X):
            off = off[0]
            if isinstance(off, str):
                off = [int(e) for e in off[1:-1].split(",")]
            ind = np.random.randint(0, len(off))
            _start = problem.data.get_ship(off[max(0, ind - 1)]).times[0]
            _end = problem.data.get_ship(off[min(len(off)-1, ind + 1)]).times[-1]

            avail = set(problem.data.ships_in_working_area(start=_start, end=_end)) - set(
                off[:ind] + off[ind + 1:]) - set([0])

            choice = np.random.randint(0, 3)
            if choice == 0:
                # Grow
                sub_seq = np.array(list(avail))[np.random.permutation(len(avail))[:1]]
                X[k] = str(off[:ind] + sub_seq.tolist() + off[ind + 1:])

            elif choice == 1:
                # Shrink
                pass
                # if len(off) > 1:
                #     X[k] = str(off[:ind] + off[ind + 1:])
            else:
                # Permute
                sub_seq = np.array(list(avail))[np.random.permutation(len(avail))[:1]]
                X[k] = str(off[:ind] + sub_seq.tolist() + off[ind + 1:])

        return pop.new("X", X)

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
            H.add(val.X[0])

    for i, (val,) in enumerate(pop.get("X")):
        if val in H:
            is_duplicate[i] = True
        H.add(val)

    return is_duplicate


if __name__ == "__main__":
    problem = load()

    np.random.seed(1)

    my_problem = MOOProblem(problem)

    algorithm = NSGA2(
        pop_size=100,
        sampling=MySampling(),
        crossover=MyCrossover(),
        mutation=MOOMutation(),
        eliminate_duplicates=func_is_duplicate)

    res = minimize(my_problem,
                   algorithm,
                   ('n_gen', 230),
                   seed=1,
                   verbose=True)
    I = np.argsort(-res.F[:, 0])
    for i in I:
        print(f"{int(-res.F[i, 0])} - {res.F[i, 1]} -- {res.X[i]}")
    # print(res.F[I])
    # print(res.X[I])
