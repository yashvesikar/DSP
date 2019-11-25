def generate_ampl_code(seq):

    for i, j in enumerate(seq):
        print(f"let seq[{i+1}] := shipo[{j}];")

def check_pf(distances, T):
    """
    Prints the solution distances to the pareto-solutions with colors
    Prints the violation of the distances vs the pareto-solutions,
    (-) violation indicates solutions better than the pf found
    (+) violation indicated the solutions are worse than the pf

    :param distances: List of distances of solutions
    :param T: Time window
    :return: None if incorrect time window specified
            float if Violation exists
    """

    def green(x): return f"\033[92m {x}\033[00m"
    def red(x): return f"\033[91m {x}\033[00m"

    if T == 4:
        pf = [41.1, 42.8, 43.4, 46.6, 47.3, 48.7, 50.5, 53.8, 60.0, 70.1, 80.8, 90.1, 100.6, 115.0, 128.2]
    elif T == 6:
        pf = [29.6, 41.3, 42.9, 43.5, 46.6, 47.3, 48.7, 50.5, 53.8, 55.4, 60.5, 70.6, 81.4, 89.5, 99.5, 110.6, 124.4,
              176.2, 190.4, 209.8]
    elif T == 8:
        pf = [29.6, 41.3, 42.9, 43.5, 46.2, 47.3, 48.7, 50.3, 51.0, 52.5, 55.8, 57.3, 61.9, 67.2, 76.8, 81.5, 86.7,
              94.4, 107.2, 112.8, 122.2, 133.2, 147.1, 159.4, 173.3]
    else:
        print(f"Pareto-front for T = {T} not available ")
        return

    violation = 0
    for i, s in enumerate(distances):
        s = round(s, ndigits=1)
        violation += s - pf[i]
        sol = green(f'sol: {s}') if s <= pf[i] else red(f'sol: {s}')
        pf_sol = green(f'pf: {pf[i]}') if pf[i] <= s else red(f'pf: {pf[i]}')
        print(f"Length: {i+1} - {sol} - {pf_sol}")
    print(f"Violation: {violation}")
    return violation

if __name__ == "__main__":
    distances = [29.605, 41.287, 41.864, 43.474]
    check_pf(distances, T=6)
