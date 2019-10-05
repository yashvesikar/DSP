from math import isclose

import numpy as np
from amplpy import AMPL, Environment
import unittest

from dsp.Problem import Problem
from dsp.Solver import DPSolver

x_data = np.genfromtxt("../data/x.csv", delimiter=",")
y_data = np.genfromtxt("../data/y.csv", delimiter=",")

xy_data = np.stack([x_data, y_data], axis=2)

P = Problem(xy_data)

class MyTestCase(unittest.TestCase):

    def test_ampl_installation(self):
        from amplpy import AMPL, Environment
        ampl = AMPL(Environment('/home/yash/amplide.linux64/'))  # Will raise execption if ampl installation is not correct

    @staticmethod
    def evaluate(seq):
        S = DPSolver(P, seq=[0] + seq + [0])
        sol = S.solve(return_distance=True)
        F = sol[1] if sol else None
        ampl_F = ampl_solve(seq)
        if F is None and ampl_F is None:
            return True

        elif F is not None and ampl_F is not None:
            return isclose(F, ampl_F, rel_tol=1e-5, abs_tol=0.0)

        else:
            return False


    def random(self):
        pass

    def test_provided(self):
        # seq = [15, 5, 8, 44, 38, 4, 12, 23, 61, 28]
        seq = [15, 5, 8, 44, 38, 4, 12, 23, 61, 35]
        truth = self.evaluate(seq)
        self.assertTrue(truth)

def ampl_solve(sequence):
    ampl = AMPL(Environment('/home/yash/amplide.linux64/'))  # Will raise execption if ampl installation is not correct

    # sequence = [56, 26, 33, 8, 12]
    # sequence = [15, 5, 8, 44, 38, 4, 12, 23, 61, 28]

    # Interpret the two files
    ampl.read('/home/yash/PycharmProjects/DSP/tests/dtsp_fitness_milp')

    # Set the alpha
    alpha = ampl.getParameter('alpha')
    alpha.set(len(sequence))

    # Set the sequence of ships
    ampl_sequence = []
    shipo = ampl.getParameter('shipo')
    for s in sequence:
        ampl_sequence.append(shipo[s])

    seq = ampl.getParameter('seq')
    seq.setValues(ampl_sequence)
    print(f"Seq values {seq.getValues()}")

    # Solve
    ampl.solve()

    # Get objective entity by AMPL name
    obj = ampl.getObjective('obj')

    if obj.result() == 'infeasible':
        return None

    return obj.value()


if __name__ == '__main__':
    unittest.main()
