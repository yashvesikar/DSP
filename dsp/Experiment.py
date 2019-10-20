import json
from abc import ABC, abstractmethod
import pickle
import time
from multiprocessing.pool import Pool

import numpy as np

from dsp.Problem import Problem
from dsp.Runner import SequenceSolver
from dsp.Solver import DPSolver
from dsp.Exploration import select_exploration
from dsp.Truncation import select_truncation


class Experiment(ABC):
    def __init__(self, dname):
        self.dname = dname
        self.args = None

    # def save_experiment(self, fname, args):
    #     with open(self.dname + fname, 'wb') as fp:
    #         pickle.dump(self, fp, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     with open(self.dname + fname, 'w') as fp:
    #         json.dump(args, fp)

    @abstractmethod
    def run_experiment(self, **kwargs):
        pass

    # def open_experiment(self, fname):
    #     with open(self.dname+fname, 'rb') as fp:
    #         return pickle.load(fp)




class VariableWeightDecomposition(Experiment):
    def __init__(self, dname):
        super().__init__(dname)
        self.results = None

    def run_experiment(self, fname=None, T=6, limit=1000, alpha=10, nweights=50, **kwargs):


        def solve(truncation_args):
            print(f"Starting weight: {w}")


            result = SeqSolver.sequence_search(available=truncation_args['ALPHA'],
                                               truncation_args=truncation_args)

            print(f"finished weight: {w}")
            return result

        # Time frame

        # Data
        x_data = np.genfromtxt("../data/x.csv", delimiter=",")
        y_data = np.genfromtxt("../data/y.csv", delimiter=",")

        xy_data = np.stack([x_data, y_data], axis=2)

        # Create sequence solver object
        P = Problem(xy_data, T=T)
        root = DPSolver(P, seq=[])
        SeqSolver = SequenceSolver(problem=P, root=root, height_limit=alpha)

        # Create values for sequence search
        ALPHA = set(P.in_working_area)

        weights = np.linspace(0, 1, nweights)
        truncation_args = [{
                'ALPHA': ALPHA,
                'limit': limit,
                'method': 'decomposition',
                'w': w
            } for w in weights]
        results = []
        with Pool(processes=8) as pool:
            results = pool.map(solve, truncation_args)

        # if fname:
        #     args = {
        #         "fname": fname,
        #         "T": T,
        #         "limit": limit,
        #         "alpha": alpha,
        #         "nweights": nweights,
        #     }
        #     self.save_experiment(fname=fname, args=args)
        self.results = results
        return self.results

if __name__ == "__main__":

    E = VariableWeightDecomposition(dname="/home/yash/PycharmProjects/DSP/results/exp_static_limit/exp_10k/")

    args = {
        # "fname": "decomp.pickle",
        "T": 6,
        "limit": 1000,
        "alpha": 5,
        "nweights": 50,
    }

    res = E.run_experiment(**args)