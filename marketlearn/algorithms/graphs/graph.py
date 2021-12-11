"""Implementation of Graphs using adjacency map adjacency matrix
Author: Raj Subramanian
"""
from __future__ import annotations
from itertools import permutations
from marketlearn.algorithms.graphs.base import GraphBase
from scipy import sparse
from typing import Any, Iterator
import numpy as np


class GraphAdjacencyMap(GraphBase):
    """Implementation of Graph via adjacency map"""

    class _Vertex(GraphBase.VertexBase):
        __slots__ = "_value"

        def __init__(self, value: Any):
            self._value = value
            self._index = 0

        def __hash__(self):
            return hash(id(self))

    class _Edge(GraphBase.EdgeBase):
        __slots__ = "_start", "_end", "_value"

        def __init__(self, u, v, value: Any = None):
            self._start = u
            self._end = v
            self._value = value

        def __hash__(self):
            return hash((self._start, self._end))

        def __eq__(self, other):
            if not type(other) is type(self):
                raise ("object must be of type Edge")
            return self.endpoint() == other.endpoint()

    # beginning of graph definition
    def __init__(self, directed=False):
        """Create an empty undirected graph by default"""
        self._out = {}
        self._in = {} if directed else self._out

    def is_directed(self) -> bool:
        return self._in is not self._out

    def count_vertices(self) -> int:
        """Returns count of total vertices in a graph

        Returns
        -------
        int
            [description]
        """
        return len(self._out)

    def vertices(self) -> Iterator[_Vertex]:
        """Returns iteration of vertices keys

        Yields
        -------
        Iterator[_Vertex]
            [description]
        """
        return self._out.keys()

    def count_edges(self) -> int:
        """
        Returns count of total edges in a graph
        """
        total_edges = sum(len(self._out[v]) for v in self.get_vertices())

        return total_edges if self.is_directed() else total_edges // 2

    def get_edge(self, u, v) -> _Edge:
        """
        Returns the edge between u and v, None if its non existent
        """
        return self._out[u].get(v)

    def edges(self) -> Iterator[_Edge]:
        """Returns Iteration of all unique edges in a graph

        Yields
        -------
        Iterator[_Edge]
            iterator of all unique edges in a graph
        """
        seen = set()
        for adjacent_vertex in self._out.values():
            seen.update(adjacent_vertex.values())
        yield from seen

    def degree(self, u, outgoing=True):
        """
        Returns total outgoing edges incident to vertex u in the graph
        for directed graph, optional parameter counts incoming edges
        """
        temp = self._out if outgoing else self._in

        return len(temp[u])

    def incident_edges(self, u: _Vertex, outgoing=True):
        """Returns iteration of all outgoing edges incident to vertex u
        in this graph
        for directed graph, optional paramter will receive incoming edges

        Parameters
        ----------
        u : _Vertex
            [description]
        outgoing : bool, optional, default=True
            [description]

        Yields
        -------
        [type]
            [description]
        """
        temp = self._out if outgoing else self._in

        for edge in temp[u].values():
            yield edge

    def iter_edges(self, nodelist: list, outgoing=True):
        """
        Similar to incident_edges, except
        takes a nodelist and returns all outgoing edges
        incident to vertices in nodelist
        """
        temp = self._out if outgoing else self._in

        for node in nodelist:
            for edge in temp[node].values():
                yield edge.endpoint() + (edge,)

    def insert_vertex(self, value: Any = None) -> _Vertex:
        """Inserts vertex with value into graph

        Parameters
        ----------
        value : Any, optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        count = self.count_vertices()
        u = self._Vertex(value)
        u._index += count
        self._out[u] = {}

        if self.is_directed():
            self._in[u] = {}

        return u

    def insert_edge(self, u: _Vertex, v: _Vertex, value: Any = None) -> _Edge:
        """Inserts and returns a new edge from u to v
        with value val (identifies the edge)

        Parameters
        ----------
        u : _Vertex
            [description]
        v : _Vertex
            [description]
        value : Any, optional
            [description], by default None

        Returns
        -------
        _Edge
            [description]
        """
        edge = self._Edge(u, v, value)

        self._out[u][v] = edge
        self._in[v][u] = edge

        return edge

    def get_adjacency_pairs(self):
        for e in self.edges():
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
