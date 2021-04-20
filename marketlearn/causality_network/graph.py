"""
Implementation of Graph Datastructure
"""
from itertools import permutations
from scipy import sparse
import numpy as np

# pylint: disable=invalid-name


class GraphAdMap:
    """
    Implementation of a graph using adjacency map

    For a pair of vertices (u,v), (u,z) that has an edges E, F
    is represented by {u: {v: E, z: F}}
    """

    class _Vertex:
        """
        Vertex structure for graph
        """

        __slots__ = ["_value", "_index"]

        def __init__(self, val):
            self._value = val
            self._index = 0

        def __repr__(self):
            """ "
            Allows outputting vertex representation
            """
            return """Vertex({!r})""".format(self._value)

        def __hash__(self):
            """
            Allows vertex to be a key in a dictionary
            """
            return hash(id(self))

        def get_value(self):
            """
            Returns value associated with this vertex
            """
            return self._value

    class _Edge:
        """
        Implements the edge structure that returns edge associated
        with vertex (u,v)
        """

        # Lightweight edge structure
        __slots__ = "start", "end", "value"

        def __init__(self, u, v, val):
            self.start = u
            self.end = v
            self.value = val

        def __repr__(self):
            """ "
            Allows outputting edge representation
            """
            insert = (self.start, self.end, self.value)

            return """Edge(({!r}, {!r}): {:.2f}""".format(*insert)

        def __hash__(self):
            """
            Allows edge to be a key in a dictionary
            """
            return hash((self.start, self.end))

        def endpoint(self):
            """
            Returns (u,v) as a tuple for vertices u and v
            """
            return (self.start, self.end)

        def opposite(self, u):
            """
            Returns vertex opposite of u on this edge
            """
            return self.end if u is self.start else self.start

        def get_value(self):
            """
            Returns value associated with this edge
            """
            return self.value

        def get_items(self):
            """
            Returns edge attributes as a tuple

            Helpful for visualizing nodes and their edge weights
            """
            return (self.start.value, self.end.value, self.value)

    # -- beginning of graph definition
    def __init__(self, directed=False):
        """
        Creates an empty graph undirected by default

        Graph is directed if parameter is set to True
        """
        self._out = {}
        self._in = {} if directed else self._out

    def is_directed(self):
        """
        Return True if graph is directed, False otherwise
        """
        return self._in is not self._out

    def count_vertices(self):
        """
        Returns the count of total vertices in a graph
        """
        return len(self._out)

    def get_vertices(self):
        """
        Returns iteration of vertices keys
        """
        return self._out.keys()

    def count_edges(self):
        """
        Returns count of total edges in a graph
        """
        total_edges = sum(len(self._out[v]) for v in self.get_vertices())

        return total_edges if self.is_directed() else total_edges // 2

    def get_edge(self, u, v):
        """
        Returns the edge between u and v, None if its non existent
        """
        return self._out[u].get(v)

    def get_edges(self):
        """
        Returns iteration of all unique edges in a graph
        """
        seen = set()

        for inner_map in self._out.values():
            seen.update(inner_map.values())

        return seen

    def degree(self, u, outgoing=True):
        """
        Returns total outgoing edges incident to vertex u in the graph
        for directed graph, optional parameter counts incoming edges
        """
        temp = self._out if outgoing else self._in

        return len(temp[u])

    def iter_incident_edges(self, u, outgoing=True):
        """
        Returns iteration of all outgoing edges incident to vertex u in this graph
        for directed graph, optional paramter will receive incoming edges
        """
        temp = self._out if outgoing else self._in

        for edge in temp[u].values():
            yield edge

    def iter_edges(self, nodelist: list, outgoing=True):
        """
        Similar to iter_incident_edges, except
        takes a nodelist and returns all outgoing edges
        incident to vertices in nodelist
        """
        temp = self._out if outgoing else self._in

        for node in nodelist:
            for edge in temp[node].values():
                yield edge.endpoint() + (edge,)

    def insert_vertex(self, val=None):
        """
        Inserts and returns a new vertex with value val
        """
        count = self.count_vertices()
        u = self._Vertex(val)
        u._index = count if count != 0 else u._index
        self._out[u] = {}

        if self.is_directed():
            self._in[u] = {}

        return u

    def insert_edge(self, u, v, val=None):
        """
        Inserts and returns a new edge from u to v
        with value val (identifies the edge)
        """
        edge = self._Edge(u, v, val)

        self._out[u][v] = edge
        self._in[v][u] = edge

        return edge

    def get_adjacency_pairs(self):
        for e in self.get_edges():
            u, v = e.endpoint()
            yield (u._index, v._index)

    def get_adjacency_matrix(self) -> np.ndarray:
        """Generates a adjacency matrix from graph

        :return: adjacency matrix
        :rtype: np.ndarray
        """
        pairs = dict.fromkeys(self.get_adjacency_pairs(), 1)
        vertices = range(self.count_vertices())
        n = len(vertices)
        arr = np.zeros((n, n))
        for p in permutations(vertices, 2):
            i, j = p
            arr[i, j] = pairs.get((i, j), 0)
        return sparse.csc_matrix(arr)
