from amplpy import AMPL
ampl = AMPL()

# Interpret the two files
ampl.read('dtsp_fitness_milp')

alpha = ampl.getParameter('alpha')
alpha.set(5)
seq = ampl.getParameter('seq')
seq.setValues([25, 11, 16, 4, 5])
print(f"Seq values {seq.getValues()}")
# Solve
ampl.solve()

# Get objective entity by AMPL name
obj = ampl.getObjective('obj')
# Print it
print("Objective is:", obj.value())