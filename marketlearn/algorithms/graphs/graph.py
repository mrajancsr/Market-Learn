"""Implementation of Graphs using adjacency list, edge list, adjacency map and adjacency matrix
Author: Raj Subramanian
Date: July 20, 2020
"""
from graphs.base import GraphBase
from linked_lists.linked_collections import PositionalList

class GraphEdgeList(GraphBase):
    """implementation of Graph via EdgeList
    Args: 
    None
    Returns: 
    Graph in a list format
    """
    class _Vertex(GraphBase.VertexBase): 
        def __init__(self, x, pos=None):
            self._value = x 
            self._pos = pos 
    
    class _Edge(GraphBase.EdgeBase): 
        def __init__(self, u, v, x=None, pos=None):
            self._start = u 
            self._end = v 
            self._value = x 
            self._pos = pos

        def __hash__(self):
            return hash((self._start, self._end))

        def __eq__(self, other):
            if not type(other) is type(self): raise("object must be of type Edge")
            return self.endpoint() == other.endpoint()

    # beginning of graph definition
    def __init__(self, directed=False):
        self._V = PositionalList()
        self._E = PositionalList()
        self._in = PositionalList() if directed else self._V
    
    def is_directed(self):
        return self._in is not self._V
    
    def count_vertices(self):
        return len(self._V)
    
    def add_vertex(self, x):
        u = self._Vertex(x)
        p = self._V.add_first(u)
        u._pos = p 
        return u
    
    def add_edge(self, u, v, x):
        e = self._Edge(u, v, x)
        p = self._E.add_first(e)
        e._pos = p
        return e
    
    def get_vertices(self):
        """returns the vertices of the graph"""
        return set(self._V)

    def get_edges(self):
        """return iteration of all unique edges in a graph"""
        return set(self._E)







g = GraphEdgeList()
u = g.add_vertex("raju")
v = g.add_vertex("prema")
print(u), print(v)
e = g.add_edge(u, v, 22)
e2 = g.add_edge(u, v, 22)
print(g.get_vertices())
