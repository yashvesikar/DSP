from collections import deque
import numpy as np
import itertools
import matplotlib.pyplot as plt

from pymoo.util.normalization import normalize

def decomposition_truncation(Q, w=0.5, **kwargs):
    data = {}
    if kwargs.get('limit'):
        limit = kwargs['limit']
    else:
        limit = 1e4

    d = np.array([q.solution[0] for q in Q])
    t = np.array([q.solution[2] for q in Q])

    d_norm = normalize(d)
    t_norm = normalize(t, x_min=0, x_max=72)

    F = (w * d_norm) + ((1-w) * t_norm)
    # F = d * (t/72)

    I = np.argsort(F)[:limit]
    Q = deque([Q[i]for i in I])

    # F = lambda x: x.states[-1].distances[0] * (x.states[-1].schedule[0] / 72)
    # Q = sorted(Q, key=criteria)

    # if len(Q) > limit:
    #     Q = Q[:limit]

    Q.append(None)  # Mark the end of a level
    return deque(Q), data

def time_truncation(Q, **kwargs):
    data = {}
    if kwargs.get('limit'):
        limit = kwargs['limit']
    else:
        limit = 1e4

    Q = sorted(Q, key=lambda x: x.states[-1].schedule[0])
    if len(Q) > limit:
        Q = Q[:limit]
    Q.append(None)
    return deque(Q), data


def distance_truncation(Q, **kwargs):
    data = {}
    if kwargs.get('limit'):
        limit = kwargs['limit']
    else:
        limit = 1e4

    criteria = lambda x: x.states[-1].distances[0]
    Q = sorted(Q, key=criteria)

    if len(Q) > limit:
        Q = Q[:limit]

    Q.append(None)  # Mark the end of a level
    return deque(Q), data


def nds_truncation(Q, type='time', **kwargs):
    data = {}

    if kwargs.get('limit'):
        limit = kwargs['limit']
    else:
        limit = 1e4

    # Construct the objective value data set
    F = []
    ind = []
    for i, m in enumerate(Q):
        assert m.feasible is True and len(m.solution) == 3
        if type == 'alpha':
            F.append((-m.solution[1], m.solution[0]))  # order by alpha and dist
        elif type == 'harbor':
            previous_ship_distance = m.states[-2].distances[m.states[-1].indexes[-1]]
            F.append((previous_ship_distance, m.solution[0]))
        else:
            F.append((m.solution[2], m.solution[0]))  # order by time and dist
        ind.append(i)

    ind = np.array(ind)
    F = np.array(F)
    if len(ind) < limit:
        limit = len(ind)
    indexes, fronts = nds_2d(F, limit=limit)  # Min-Min 2D NDS
    q = []
    for j in ind[list(itertools.chain(*indexes))]:
        q.append(Q[j])

    q.append(None)
    Q = deque(q)
    return Q, data


def nds_2d(F, **kwargs):
    mask = np.lexsort((F[:, 0], F[:, 1]))  # Sort by last index (dist), break ties on other one
    fronts = []
    count = 0
    indexes = []

    # Used to get metrics on number of fronts and size of splitting front
    add = True
    last_front_size = 0
    while count < kwargs['limit']:
        front = [F[mask[0]]]

        indicies = [mask[0]]
        minimum = front[0][0]
        count += 1
        new_mask = []
        for i, ind in enumerate(mask):
            if i == 0:
                continue
            if count > kwargs['limit']:
                add = False
                last_front_size = len(front)

            if F[ind][0] < minimum:
                if add:
                    front.append(F[ind])
                    indicies.append(ind)
                    minimum = F[ind][0]
                    count += 1
                else:
                    last_front_size += 1

            else:
                if add:
                    new_mask.append(ind)

        mask = new_mask
        fronts.append(front)
        indexes.append(indicies)
        if add is False:
            break

    return indexes, fronts


# def statistical_limit(Q):
#     import matplotlib.pyplot as plt
#     plt.hist(Q)
#     plt.show()
#     input()
#     return Q[:10000]


def exhsaustive_truncation(Q, **kwargs):
    data = {}
    Q.append(None)
    return Q, data


def select_truncation(method):

    methods = {"exhaustive": exhsaustive_truncation,
               "time": time_truncation,
               "distance": distance_truncation,
               "decomposition":decomposition_truncation,
               "nds": nds_truncation}
    if methods.get(method):
        return methods.get(method)




if __name__ == "__main__":
    from pymoo.util.nds.naive_non_dominated_sort import naive_non_dominated_sort
    import matplotlib.pyplot as plt

    np.random.seed(10)
    vals = np.random.rand(50, 2)
    indicies, fronts = nds_2d(vals, limit=50)
    pymoo_fronts = naive_non_dominated_sort(vals)
    print(indicies[0])
    print(pymoo_fronts[0])

    for x, y in zip(indicies, pymoo_fronts):
        if sorted(x) == y:
            print("True")
        else:
            print("False")

    plt.scatter(x=np.array(fronts[0])[:, 0],
                y=np.array(fronts[0])[:, 1])
    plt.scatter(x=vals[pymoo_fronts[0], 0],
                y=vals[pymoo_fronts[0], 1])

    plt.show()
