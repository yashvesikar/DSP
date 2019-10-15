from collections import deque
import numpy as np
import itertools
import matplotlib.pyplot as plt


def distance_truncation(Q, **kwargs):
    data = {}
    if kwargs.get('limit'):
        limit = kwargs['limit']
    else:
        limit = 1e4
    Q = sorted(Q, key=lambda x: x.states[-1].distances[0])

    d = [q.states[-1].distances[0] for q in Q]
    min_d, max_d = d[0], d[-1]
    bins = np.linspace(min_d, max_d, 30)

    if len(Q) > limit:
        # Q = statistical_limit(Q)
        Q = Q[:limit]

    d2 = [q.states[-1].distances[0] for q in Q]

    plt.hist([d, d2], bins=bins)
    plt.title(f"level {len(Q[0].states)-2} - count {len(Q)}")

    plt.show()
    Q.append(None)  # Mark the end of a level
    return deque(Q), data



def nds_truncation(Q, **kwargs):

    data = {}

    if kwargs.get('limit'):
        limit = kwargs['limit']
    else:
        limit = 1e4

    # S
    if kwargs.get('type') == 'alpha':
        type = 'alpha'
    else:
        type = 'time'

    ######## Initial histogram bins
    d = [q.states[-1].distances[0] for q in Q]
    min_d, max_d = d[0], d[-1]
    bins = np.linspace(min_d, max_d, 30)
    ########
    # Construct the objective value data set
    F = []
    ind = []
    for i, m in enumerate(Q):
        assert m.feasible is True and len(m.solution) == 3
        if type == 'alpha':
            F.append((-m.solution[1], m.solution[0]))  # order by alpha and dist
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

    ###### After truncation Histogram
    d2 = [l.states[-1].distances[0] for l in q]
    plt.hist([d, d2], bins=bins)
    plt.title(f"NDS level {len(Q[0].states)-2} - count {len(Q)}")
    plt.show()
    ######

    q.append(None)
    Q = deque(q)
    return Q, data


def nds_2d(F, **kwargs):

    mask = np.lexsort((F[:, 0], F[:, 1]))  # Sort by last index (dist), break ties on other one
    fronts = []
    count = 0
    indexes = []
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
                break

            if F[ind][0] < minimum:
                front.append(F[ind])
                indicies.append(ind)
                minimum = F[ind][0]
                count += 1
            else:
                new_mask.append(ind)

        mask = new_mask
        fronts.append(front)
        indexes.append(indicies)

    return indexes, fronts


def statistical_limit(Q):
    import matplotlib.pyplot as plt
    plt.hist(Q)
    plt.show()
    input()
    return Q[:10000]




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