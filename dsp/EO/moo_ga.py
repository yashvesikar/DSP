import copy
from math import ceil

import numpy as np

from dsp.HTSolver import HeuristicTreeSolver
from dsp.Problem import load_problem, ShipProblem, load_data, MOOProblem
from dsp.Solver import solve_sequence
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting





class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        truncation_args = {'limit': 25, 'method': "distance"}
        level_order_solver = HeuristicTreeSolver(problem=problem, max_depth=1000, truncation_args=truncation_args)
        ret = level_order_solver.solve()["selected"]
        n_each_seq = ceil(n_samples / len(ret))

        X = []
        for level in ret:
            seq = [e.seq[1:-1] for e in level[:n_each_seq]]
            X.extend(seq)

        X = np.array(X, dtype=np.object)[:, None]
        return X


def crossover(p_a, p_b, s_a):
    I = np.random.permutation(len(p_a))
    for i in I:
        J = np.random.permutation(len(p_b))

        for j in J:
            prefix, suffix = p_a[:i], p_b[j:]
            suffix = [s for s in suffix if s not in prefix]
            #return prefix + suffix

            if len(suffix) > 0:

                solver = copy.copy(s_a)
                solver.feasible = False
                solver.states = s_a.states[:i + 1]
                solver.seq = s_a.seq[:i + 1]

                solver.next(suffix[0])
                solver.feasible = len(solver.states[-1].times) > 0

                if solver.feasible:
                    return prefix + suffix

    return p_a.tolist()


class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def do(self, problem, pop, parents, algorithm=None, **kwargs):
        solvers = algorithm.pop.get("solver")

        off = pop.new(len(parents) * self.n_offsprings)

        X = []
        for k, (a, b) in enumerate(parents):
            s_a, s_b = solvers[a, 0], solvers[b, 0]
            p_a, p_b = s_a.seq[1:-1], s_b.seq[1:-1]
            off[2 * k].X =  [crossover(p_a, p_b, s_a)]
            off[2 * k + 1].X = [crossover(p_b, p_a, s_b)]

        return off




class MOOMutation(Mutation):

    def do(self, problem, pop, **kwargs):

        for k, off in enumerate(pop):
            X = off.X[0]

            ind = np.random.randint(0, len(X))
            _start = problem.get_ship(X[max(0, ind - 1)]).times[0]
            _end = problem.get_ship(X[min(len(X)-1, ind + 1)]).times[-1]

            avail = set(problem.ships_in_working_area(start=_start, end=_end)) - set(
                X[:ind] + X[ind + 1:]) - set([0])

            choice = np.random.randint(0, 3)
            if choice == 0:
                # Grow
                sub_seq = np.array(list(avail))[np.random.permutation(len(avail))[:1]]
                off.X[0] = X[:ind] + sub_seq.tolist() + X[ind + 1:]

            elif choice == 1:
                # Shrink
                pass
                # if len(off) > 1:
                #     X[k] = str(off[:ind] + off[ind + 1:])
            else:
                # Permute
                sub_seq = np.array(list(avail))[np.random.permutation(len(avail))[:1]]
                off.X[0] = X[:ind] + sub_seq.tolist() + X[ind + 1:]

        return pop

def my_callback(algorithm):
    disp = algorithm.func_display_attrs(algorithm.problem, algorithm.evaluator, algorithm, algorithm.pf)
    disp = [e for e in disp if e[0] not in ["igd", "gd"]]

    pop = algorithm.pop
    pf = algorithm.problem.pareto_front()

    max_n_ships = max([len(ind.X[0]) for ind in pop])
    disp.append(('n_ships', max_n_ships, 5))

    error = 0

    algorithm._display(disp)


def func_is_duplicate(pop, *other, **kwargs):
    if len(other) == 0:
        return np.full(len(pop), False)

    # value to finally return
    is_duplicate = np.full(len(pop), False)

    H = set()
    for e in other:
        for val in e:
            H.add(str(val.X[0]))

    for i, (val,) in enumerate(pop.get("X")):
        if str(val) in H:
            is_duplicate[i] = True
        H.add(str(val))

    return is_duplicate


def solve_moo(T=6, verbose=False):
    data, pf = load_data(pf=True, T=T)
    my_problem = MOOProblem(xy_data=data, T=T, pf=pf)

    algorithm = NSGA2(
        pop_size=100,
        sampling=MySampling(),
        crossover=MyCrossover(),
        mutation=MOOMutation(),
        callback=my_callback,
        eliminate_duplicates=func_is_duplicate)

    res = minimize(my_problem,
                   algorithm,
                   ('n_gen', 250),
                   seed=1,
                   verbose=False)

    I = np.argsort(-res.F[:, 0])
    for i in I:
        print(f"{int(-res.F[i, 0])} - {res.F[i, 1]} -- {res.X[i]}")

    results = [i.data['solver'][0] for i in res.opt]
    return {"solvers": results}

if __name__ == "__main__":
    np.random.seed(1)
    solve_moo(T=6, verbose=True)