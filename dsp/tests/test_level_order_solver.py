import numpy as np

from dsp.Problem import load_problem
from dsp.Runner import SequenceSolver
from dsp.Solver import solve_sequence
import unittest

class TestLevelOrderSolver(unittest.TestCase):

    def test_distance(self):
        # np random seed = 10
        data = np.load('level_order_solver.npy', allow_pickle=True)
        problem = load_problem(T=6)
        truncation_args = {'limit': 1000, 'method': "distance"}
        optimizer = SequenceSolver(problem=problem, max_height=10, truncation_args=truncation_args)
        results = optimizer.solve()

        for i, j in enumerate(data):
            d = j[0]
            seq = j[1:][0]
            solver = results['solvers'][i]

            self.assertTrue(solver.result.feasible)
            self.assertAlmostEqual(d, solver.result.distance)
            self.assertEqual(seq, solver.result.seq)
