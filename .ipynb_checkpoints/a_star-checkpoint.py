import random

coords = dict([(1, (-40.49770830225964, -24.028587264111685)),
(2, (-7.628608961568382, -13.177137821152122)),
(3, (-64.26930584682951, -50.52795519558309)),
(4, (-51.28608407537351, -56.12389133156939)),
(5, (-32.7327520932375, -51.52047020179791)),
(6, (-56.29339091282749, -15.734844568977344)),
(8, (-51.599379105551066, -21.41681727120111)),
(9, (-43.95195088489723, -44.83584488969718))])

def Generate_edges(size, connectedness):
    """
    DO NOT EDIT THIS FUNCTION
    Generates directed edges between vertices to form a DAG
    :return: A generator object that returns a tuple of the form (source ID, destination ID)
    used to construct an edge
    """

    assert connectedness <= 1
    random.seed(10)
    for i in range(size):
        for j in range(i + 1, size):
            if random.randrange(0, 100) <= connectedness * 100:
                yield f'{i} {j}'


# Custom Graph error
class GraphError(Exception): pass


class Vertex:
    """
    Class representing a Vertex in the Graph
    """
    __slots__ = ['ID', 'index', 'visited', 'f', 'g', 'h', 'coords', 'parent']

    def __init__(self, ID, index, parent=None):
        """
        Class representing a vertex in the graph
        :param ID : Unique ID of this vertex
        :param index : Index of vertex edges in adjacency matrix
        """
        self.ID = ID
        self.index = index  # The index that this vertex is in the matrix
        self.visited = False
        self.coords = coords[ID]
        self.f = 0
        self.g = 0
        self.h = 0
        self.parent = None

    def __hash__(self):
        return hash(self.coords)

    def __repr__(self):
        return f"Vertex: {self.ID}"

    __str__ = __repr__

    def __eq__(self, other):
        """
        DO NOT EDIT THIS METHOD
        :param other: Vertex to compare
        :return: Bool, True if same, otherwise False
        """
        return self.ID == other.ID and self.index == other.index

    def __lt__(self, other):
        return self.f < other.f

    def out_degree(self, adj_matrix):
        """
        :param adj_matrix: matrix looked up from Graph object
        Returns the number of outgoing edges of the vertex
        :return: out degree of Vertex
        """
        return sum(1 for i in adj_matrix[self.index] if i is not None)

    def in_degree(self, adj_matrix):
        """
        :param adj_matrix: matrix looked up from Graph object
        Returns the number of incoming edges of the vertex
        :return: in degree of Vertex
        """
        return sum(1 for i in range(len(adj_matrix)) if adj_matrix[i][self.index] is not None)


class Graph:
    """
    Graph Class ADT
    """

    def __init__(self, iterable=None):
        """
        DO NOT EDIT THIS METHOD
        Construct a random Directed Graph
        :param size: Number of vertices
        :param: iterable: iterable containing edges to use to construct the graph.
        """
        self.id_map = {}
        self.size = 0
        self.matrix = []
        self.iterable = iterable
        self.construct_graph()
        if hasattr(iterable, 'close'):
            iterable.close()

    def get_vertex(self, ID):
        """
        Returns the Vertex with the specified ID
        :param ID: ID of Vertex to search for
        :return: Vertex or None
        """
        if ID in self.id_map:
            return self.id_map[ID]

    def get_edges(self, ID):
        """
        Returns a list of edge objects given the ID of a vertex
        :param ID: ID of a vertex
        :return:
        """
        vertex = self.get_vertex(ID)
        if vertex:
            return {i for i in self.matrix[vertex.index] if i is not None}
        return set()
    def construct_graph(self):
        """
        This function constructs the graph
        Called within the class constructor
        Uses insert_edge method to insert an edge
        :return: None
        """
        if self.iterable is None:
            raise GraphError
        for edge in self.iterable:
            source, destination = list(map(int, edge.split()))
            self.insert_edge(source, destination)

    def insert_edge(self, source, destination):
        """
        Insert an edge into the graph.
        Creates a vertex if it does not exist and adds it to the self.adj_list
        :param source: Source ID of the edge
        :param destination: Destination ID of the edge
        :return: None
        """

        # If the source or destination does not exist then the edge can not exist
        # Create vertices
        if source not in self.id_map:
            for r in self.matrix:
                r.append(None)
            self.id_map[source] = Vertex(source, self.size)
            self.size += 1
            self.matrix.append([None for _ in range(self.size)])

        if destination not in self.id_map:
            for r in self.matrix:
                r.append(None)
            self.id_map[destination] = Vertex(destination, self.size)
            self.size += 1
            self.matrix.append([None for _ in range(self.size)])

        i, j = self.id_map[source], self.id_map[destination]
        self.matrix[i.index][j.index] = destination

    def a_star(self, start, end):
        import heapq

        op = []
        closed = set()

        # put the start node on the open list ( leave its f value at 0)
        curr = self.get_vertex(start)
        op.append(curr)
        heapq.heapify(op)

        end_node = self.get_vertex(end)
        while op:
            curr = heapq.heappop(op)
            closed.add(curr.ID)

            if curr.ID == end:
                path = []
                current = curr
                while current is not None:
                    path.append(current.ID)
                    current = current.parent
                return path[::-1]


            adj = self.get_edges(curr.ID)

            for child in adj:
                if child in closed:
                    continue
                else:
                    child_node = self.get_vertex(child)
                    child_node.g = curr.g + 1
                    child_node.h = abs(child_node.coords[0] - end_node.coords[0]) + abs(child_node.coords[1] - end_node.coords[1])
                    child_node.f = child_node.g + child_node.h

                    if child in op:
                        if op and child.g > op[0].g:
                            continue
                        else:
                            child.parent = curr
                            heapq.heappush(child)

    def visualize(self):
        import matplotlib.pyplot as plt
        x = [v[0] for k, v in coords.items()]
        y = [v[1] for k, v in coords.items()]
        pid = [k for k, v in coords.items()]

        fig, ax = plt.subplots()
        ax.plot(x, y, ls="", marker="o")
        for xi, yi, pidi in zip(x, y, pid):
            ax.annotate(str(pidi), xy=(xi, yi))

        plt.show()









if __name__ == "__main__":

    filename = open("Graphs/Project8Solution/test_search_simple.txt", 'r')

    stu = Graph(iterable=filename)
    print(stu.id_map)
    stu.visualize()
    stu.a_star(4, 6)

    # all_paths = [[1, 3, 5, 6, 9], [1, 3, 6, 9]]
