"""Implementation of Graphs using adjacency list, edge list, adjacency map and adjacency matrix
Author: Raj Subramanian
Date: July 20, 2020
"""
from __future__ import annotations
from marketlearn.algorithms.graphs.base import GraphBase
from typing import Any


class GraphAdjacencyMap(GraphBase):
    """Implementation of Graph via adjacency map"""

    class _Vertex(GraphBase.VertexBase):
        __slots__ = "_value"

        def __init__(self, value):
            self._value = value

        def __hash__(self):
            return hash(id(self))

    class _Edge(GraphBase.EdgeBase):
        __slots__ = "_start", "_end", "_value"

        def __init__(self, u: _Vertex, v: _Vertex, value: Any = None):
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
        self._in if directed else self._out

    def is_directed(self) -> bool:
        return self._in is not self._out


g = GraphAdjacencyMap()
