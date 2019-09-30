import mosek
from mosek.fusion import *

with Model('DSP') as M:
    import numpy as np

    T = 6  # Time horizon (hours), initial time stage = 0. time period [0, T]

    n = 64  # Total number od ships visiting the work area during [0, T]

    w = 5 / 60  # Time slot width in hours
    m = np.floor((T / w) + 0.1)  # The number of time slots in T, with time slot size w

    p = 3 / 60  # The service time in hours needed per ship. NOTE: we require p < w
    v = 46.3  # The speed of the service vessel in km/h


    # Data
    # x_data = np.genfromtxt("data/x.csv", delimiter=",")[:int(m) + 1, :]
    x_data = np.genfromtxt("data/x.csv", delimiter=",")
    y_data = np.genfromtxt("data/y.csv", delimiter=",")
    # y_data = np.genfromtxt("data/y.csv", delimiter=",")[:int(m) + 1, :]

    # Repair missing values
    x_data[np.isnan(x_data)] = 0
    y_data[np.isnan(y_data)] = 0

    # Create more intelligent data representation
    # axis 0: time slot
    # axis 1: ship
    # axis 2: [x, y] coordinate
    xy_data = np.stack([x_data, y_data], axis=2)

    first = np.full(n, m, dtype=np.int)  # First slot when ship i is in the work area
    last = np.full(n, 0, dtype=np.int)  # Lasst slot when ship i is in the work area

    # For loop to fill out first, and last array
    for s in range(n):
        c = np.all(xy_data[:, s] != 0, axis=1)
        non_zero = np.where(c)[0]
        if len(non_zero):
            first[s] = non_zero[0]
            last[s] = non_zero[-1]

    # Loop to reorganize data and find number of eligible ships
    j = -1
    x_data_mask = np.zeros(x_data.shape)
    y_data_mask = np.zeros(y_data.shape)
    first_mask = np.zeros(n, dtype=np.int)
    last_mask = np.zeros(n, dtype=np.int)
    shipn = np.full(n, np.nan)
    shipo = np.full(n, np.nan)

    for jj in range(n):
        if first[jj] <= last[jj]:
            j += 1
            for ii in range(int(m)):
                x_data_mask[ii, j] = x_data[ii, jj]
                y_data_mask[ii, j] = y_data[ii, jj]
                first_mask[j] = first[jj]
                last_mask[j] = last[jj]
                shipn[j] = jj
                shipo[jj] = j
    # Debugging
    # fl = np.column_stack([first_mask, last_mask])
    # on = np.column_stack([shipn, shipo])
    n = j  # Number of eligible ships
    print(f"Number of eligible ships: {n}")
    print(f"Number of time slots: {m}")

    ############## begin test case: note the renumbering of ships after schrinking ###
    # for fittness evaluation, enter here values alpha and seq(*):

    # test case, number of ships to be visited during [0,T];
    alpha = 5 # input for fitness enaluation: number of ships to be visited during [0,T]; for a single optimixation fix alpha, 0<alpha<=n

    seq = np.arange(alpha)  # input for fitness enaluation: sequence of ships to be visited; seq defines one member of the population
    # seqk = np.arange(n)  # time slots of the sequence of ships to be visited in seq

    seq[0] = shipo[56]  # temporary sequence for testing
    seq[1] = shipo[26]  # temporary sequence for testing
    seq[2] = shipo[33]  # temporary sequence for testing
    seq[3] = shipo[8]  # temporary sequence for testing
    seq[4] = shipo[12]  # temporary sequence for testing


    J = np.arange(alpha)
    S0 = np.arange(m)
    S = [None] * J.shape[0]
    for i, ii in enumerate(J):
        S[i] = np.arange(first_mask[seq[ii]], last_mask[seq[ii]])

    ### end test case ###

    # z = M.variable("z", [len(J), m, m], Domain.binary())
    z0 = M.variable('z0', [len(S[0])], Domain.binary())
    zn = M.variable('zn', [len(S[-1])], Domain.binary())

    # mosek.fusion.BaseVariable.getND(z0)


    # init
    M.constraint('init', Expr.sum(z0), Domain.equalsTo(1))

    # finish constraint
    M.constraint('finish', Expr.sum(zn), Domain.equalsTo(1))

    # intermediate travel constraint : flow control
    # for ii in J:
    #     for kk in S[ii]:
    #         if ii == 1:
    #             z0
    # M.constraint()

    sub = [[] for i in range(len(J) - 1)]
    sub.extend([[], []])
    MM = []
    for ii in J[:-1]:
        ship = []
        for kk in S[ii]:
            row = []
            for ll in S[ii + 1]:
                if ii < alpha and ll > kk:
                    row.append(1)
                    # sub[ii].append(ii)
                    # sub[-2].append(kk)
                    # sub[1].append(ll)
                else: row.append(0)
            ship.append(row)
        MM.append(ship)

    z = NDSparseArray(dims=len(J)-1, sub=sub, float=[1 for _ in range(len(sub))])

    #             z.append(1 if ii < alpha and ll > kk else 0)
