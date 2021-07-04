"""Module replicates the paper "A review of two decates of correlation, 
hierarchies, networks and clustering in financial markets using minimum spanning trees
Author: Rajan Subramanian
Date: 07/16/2020
"""

import numpy as np
import yfinance as yf
import heapq
from itertools import combinations


class PriorityQueue:
    """Implementation of a priority queue using Heaps"""

    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        """pushes an item with a given priority
        Takes O(floor(logn)) time
        Args:
        item:   the item we want to store
        priority:   default to min (use negative for max)
        """
        # if two items have the same priority, a secondary
        # index is added to avoid comparison failure
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        """removes the item with the highest priority
        Takes O(1) time
        """
        return heapq.heappop(self._queue)

    def __len__(self):
        return len(self._queue)

    def is_empty(self):
        return len(self) == 0


class Cluster:
    """Union find structure for use in Kruskal's algo
    to check if undirected graph contains cycle or not
    """

    # - Nested Position class to keep track of parent
    class Position:
        """Light weight structure for slots"""

        __slots__ = "_container", "_value", "_size", "_parent"

        def __init__(self, container, value):
            self._container = container
            self._value = value
            self._size = 1
            self._parent = self

        def get_value(self):
            return self._value

    def make_cluster(self, value):
        return self.Position(self, value)

    def find(self, p):
        """find group contianing p and return position of its leader"""
        if p._parent != p:
            p._parent = self.find(p._parent)
        return p._parent

    def union(self, p, q):
        """merges group containing element p and q (if distinct)"""
        a = self.find(p)
        b = self.find(q)
        if a is not b:
            if a._size > b._size:
                b._parent = a
                a._size += b._size
            else:
                a._parent = b
                b._size += a._size


class MinimumSpanningTrees:
    """Implementation of minimum spanning tree using Kruskal's algorithm
    Args:
    None

    Attributes:
    distance:   pandas DataFrame

    Returns:
    Minimum Spanning Tree using Kruskal's aglo
    """

    # -- Nested Price Class
    class Price:
        """Get the prices from yahoo finance and calculates their distances
        Args:
        start: start date in format YYYY-MM-DD in string
        end:   end date in format YYYY-MM-DD in string

        Returns:
        distances calculated from correlation matrix (pandas object)

        Notes:
        prices are downloaded from yahoo finance using yfinance module
        log returns are calculated using the formula
            rt = log(1 + Rt) where Rt is the percentage return
        Distances are calculated using the formula:
            dij = sqrt(2 * (1 - pij))
        """

        def __init__(self, start, end):
            self.start = start
            self.end = end

        def get_prices(self, col="Adj Close"):
            """gets the adjusted close price from yahoo finance
            Args:
            col: string supported types are High, low, Adj Close
            """
            # - set ticker names here
            ticker_names = """\
            TSLA SPY MSFT MMM ABBV ABMD ACN ATVI ADBE AMD 
            AES AFL APD AKAM ALB ARE ALLE GOOG AAL AMT
            ABC AME AMGN APH ADI AMAT APTV BKR BAX BDX
            BIO BA BKNG BWA BXP BSX BMY COG COF KMX
            CBOE CE CNC CF DLTR EFX FIS GD GE GS
            """
            return yf.download(
                ticker_names, start=self.start, end=self.end, progress=False
            )[col]

        def _calculate_correlation(self, prices):
            """calculates the correlation given prices
            Args:
            prices: pandas dataframe

            Returns:
            correlation:  pandas dataframe
            """
            return np.log(1 + prices.pct_change()).corr()

        def get_distance(self):
            """Computes the distance given correlation
            dij := sqrt( 2(1 - pij) )
            """
            prices = self.get_prices()
            pairwise = self._calculate_correlation(prices)
            distance = np.sqrt(2 * (1 - pairwise))
            return distance

    # ---MST Kruskals algorithm
    def __init__(self, start, end):
        self.distance = self.Price(start, end).get_distance()

    def create_graph(self):
        """creates a graph with vertices and edges from distance
        Args:
        None

        Returns:
        graph object
        """
        g = Graph()
        share_names = iter(list(self.distance))
        vertices = iter([g.insert_vertex(v) for v in share_names])
        edges = []
        for c in combinations(vertices, 2):
            # get the vertices
            u, v = c
            # get the distance weight
            w = self.distance.loc[u.get_value(), v.get_value()]
            # create edge
            g.insert_edge(u, v, w)
        return g

    def mst_kruskal(self, g):
        """compute minimum spanning tree using kruskal's algorithm
        Args:
        g:     Graph with a adjacy map structure

        Returns:
        list of graph's edges where edges are weights
        """
        tree = []  # stores edges of a spanning tree
        pq = PriorityQueue()  # to store the minimum edges of a graph
        cluster = Cluster()
        position = {}  # map each node to partition array

        for v in g.get_vertices():
            position[v] = cluster.make_cluster(v)

        for e in g.get_edges():
            pq.push(e, e.get_value())

        size = g.count_vertices()
        while len(tree) != size - 1 and not pq.is_empty():
            weight, _, edge = pq.pop()
            u, v = edge.endpoint()
            a = cluster.find(position[u])
            b = cluster.find(position[v])
            if a != b:
                tree.append(edge)
                cluster.union(a, b)
        return tree

    def draw_graph(self, mst_tree):
        """Plots the minimum spanning tree
        Args:
        mst_tree:  list of tree objects with minimum edges from Edge Class

        Returns:
        None:       plot object containing the MST
        """
        import networkx as nx
        import matplotlib.pyplot as plt

        g = nx.Graph()
        # get the edge vertices and edge weights from mst_tree
        items = (e.get_items() for e in mst_tree)
        # add the edges in the networkx graph for plotting
        g.add_weighted_edges_from(items)
        nx.draw(g, with_labels=True)
        plt.draw()
