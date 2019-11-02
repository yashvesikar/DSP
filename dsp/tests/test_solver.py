import numpy as np

from dsp.Problem import load_problem
from dsp.Solver import solve_sequence
import unittest

class TestSolver(unittest.TestCase):

    def test_some_sequences(self):
        data = np.load('dp_solver.npy', allow_pickle=True)[:100]
        problem = load_problem(T=6)
        for i in data:
            d = i[0]
            seq = i[1:].astype(int)
            solver = solve_sequence(problem=problem, seq=seq)
            self.assertTrue(solver.result.feasible)
            self.assertAlmostEqual(d, solver.result.distance)


    def test_all_sequences(self):
        data = np.load('dp_solver.npy', allow_pickle=True)
        problem = load_problem(T=6)
        for i in data:
            d = i[0]
            seq = i[1:].astype(int)
            solver = solve_sequence(problem=problem, seq=seq)
            self.assertTrue(solver.result.feasible)
            self.assertAlmostEqual(d, solver.result.distance)
