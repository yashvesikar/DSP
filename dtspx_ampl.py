import time
from amplpy import AMPL, Environment
from multiprocessing import Pool

def ampl_solve(al):
    start = time.time()
    ampl = AMPL(Environment('/home/yash/amplide.linux64/'))  # Will raise execption if ampl installation is not correct

    # Interpret the file
    ampl.read('/home/yash/Documents/DSP/dtspx-yash.mod')

    # Set the alpha
    alpha = ampl.getParameter('alpha')
    alpha.set(al)

    # Solve
    ampl.solve()
    end = time.time()

    with open(f"results/ampl/time_{al}.txt", "w+") as fp:
        fp.write(f"Execution Time: {end-start}")

    # Get objective entity by AMPL name
    obj = ampl.getObjective('obj')

    z = ampl.getVariable('z')
    z.getValues().toPandas().to_csv(f'results/ampl/data_{al}.csv')
    with open(f"results/ampl/sol_{al}.txt", "w+") as fp:
        if obj.result() == 'infeasible':
            fp.write("Infeasible")
            return None
        else:
            fp.write(str(obj.value()))


if __name__ == "__main__":
    time_to_complete = {}
    with Pool(processes=8) as pool:
        pool.map(ampl_solve, range(2, 24))

