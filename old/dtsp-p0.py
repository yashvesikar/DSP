import numpy as np

T = 6  # Time horizon (hours), initial time stage = 0. time period [0, T]

n = 64  # Total number od ships visiting the work area during [0, T]

w = 5/60  # Time slot width in hours
m = np.floor((T/w) + 0.1)  # The number of time slots in T, with time slot size w

p = 3/60  # The service time in hours needed per ship. NOTE: we require p < w
v = 46.3  # The speed of the service vessel in km/h

alpha = n  # input for fitness enaluation: number of ships to be visited during [0,T]; for a single optimixation fix alpha, 0<alpha<=n

seq = np.arange(alpha)  # input for fitness enaluation: sequence of ships to be visited; seq defines one member of the population
# seqk = np.arange(n)  # time slots of the sequence of ships to be visited in seq

J = np.arange(alpha)

# Data
x_data = np.genfromtxt("data/x.csv", delimiter=",")[:int(m) + 1, :]
y_data = np.genfromtxt("data/y.csv", delimiter=",")[:int(m) + 1, :]

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

