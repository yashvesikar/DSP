# import unittest
#
#
# class MyTestCase(unittest.TestCase):
#
#     def test_ampl_installation(self):
#         from amplpy import AMPL, Environment
#         ampl = AMPL(Environment('/home/yash/amplide.linux64/'))  # Will raise execption if ampl installation is not correct
#
#
#     def test_provided(self):
import os

print(os.system("which python"))

from amplpy import AMPL, Environment
ampl = AMPL(Environment('/home/yash/amplide.linux64/'))  # Will raise execption if ampl installation is not correct

sequence = [56, 26, 33, 8, 12]

# Interpret the two files
ampl.read('/home/yash/PycharmProjects/DSP/tests/dtsp_fitness_milp')


# Set the alpha
alpha = ampl.getParameter('alpha')
alpha.set(5)


# Set the sequence of ships
ampl_sequence = []
shipo = ampl.getParameter('shipo')
for s in sequence:
    ampl_sequence.append(shipo[s])

seq = ampl.getParameter('seq')
seq.setValues(ampl_sequence)
print(f"Seq values {seq.getValues()}")

ampl.setOption('solver', 'mosek')
ampl.setOption('mosek_options', 'outlev=2 msk_ipar_num_threads=1')

# Solve
ampl.solve()

# Get objective entity by AMPL name
obj = ampl.getObjective('obj')
# Print it
print("Objective is:", obj.value())

# self.assertEqual(True, False)


# if __name__ == '__main__':
#     unittest.main()
