"""Module serves as a Abstration for Graph Class
Author: Rajan Subramanian
Date: July 21, 2020
"""
from abc import ABCMeta, abstractmethod


class GraphBase(metaclass=ABCMeta):
    """Abstract Base Class representing Graph Structure"""

    class VertexBase:
        """Abstraction representing vertex of a graph"""

        def value(self):
            """return value stored in this vertex"""
            return self._value

        def __repr__(self):
            return """Vertex({!r})""".format(self._value)

    class EdgeBase:
        """Abstration represent Edge of a graph"""

        def __repr__(self):
            return """Edge(({!r}, {!r}): {!r}""".format(
                self._start, self._end, self._value
            )

        def endpoint(self):
            """return (u,v) as a tuple for vertices u and v"""
            return (self._start, self._end)

        def opposite(self, u):
            """return vertex opposite of u on this edge"""
            return self._end if u is self._start else self._start

        def value(self):
            """return value associated with this edge"""
            return self._value

        def get_items(self):
            """returns edge attributes as a tuple
            Helpful for visualizing nodes and their edge weights"""
            return (self._start._value, self._end._value, self._value)

    @abstractmethod
    def is_directed(self):
        pass

    @abstractmethod
    def count_vertices(self):
        pass

    @abstractmethod
    def vertices(self):
        pass

    @abstractmethod
    def count_edges(self):
        pass

    @abstractmethod
    def get_edge(self, u, v):
        pass

    @abstractmethod
    def edges(self):
        pass

    @abstractmethod
    def degree(self, u, outgoing=False):
        """return number of outgoing edges incident to vertex u
        in the graph.  For directed graph, optional parameter counts
        incoming edges
        """
        pass

    @abstractmethod
    def insert_vertex(self, u, val):
        """insert and return a new vertex with value val"""
        pass

    @abstractmethod
    def insert_edge(self, u, v, val):
        """insert and return a new edge from vertex u to v with value val
        (identifies the edge)"""
        pass
