# Homework 1: Coding
# Author: Rajan S
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, List, Tuple

import numpy as np


# Utility Classes for graph
@dataclass
class Vertex:
    value: str
    index: int = 0

    def __hash__(self):
        return hash(id(self))


@dataclass
class Edge:
    start: str
    end: str

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __eq__(self, other) -> bool:
        return self.endpoint() == other.endpoint()

    def endpoint(self) -> Tuple[str, str]:
        return (self.start, self.end)


# utility class for chess attack
@dataclass
class Coordinate:
    xidx: int
    yidx: int
    xold: int = -1
    yold: int = -1

    def __post_init__(self):
        self.xold = self.xidx
        self.yold = self.yidx

    def __hash__(self):
        return hash((self.xidx, self.yidx))

    def __eq__(self, other):
        return self.get_coordinates() == other.get_coordinates()

    def get_coordinates(self):
        return (self.xidx, self.yidx)

    def reset_coordinates(self) -> None:
        self.xidx = self.xold
        self.yidx = self.yold

    def move_right(self) -> Coordinate:
        self.yidx += 1
        return self

    def move_up(self) -> Coordinate:
        self.xidx -= 1
        return self

    def move_down(self) -> Coordinate:
        self.xidx += 1
        return self

    def move_left(self) -> Coordinate:
        self.yidx -= 1
        return self


# ------------Priority Queue Problem --------------#########
@dataclass
class Item:
    key: Any
    value: Any

    def __lt__(self, other: Item) -> bool:
        return self.value < other.value


class EmptyException(Exception):
    pass


@dataclass
class PriorityQueue:
    """List based implementation of priority queue"""

    _data: List[Any] = field(init=False, default_factory=list)

    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return len(self) == 0

    def get_parent_index(self, child_index: int) -> int:
        """Returns index of parent of child index

        Parameters
        ----------
        child_index : int
            index of child position

        Returns
        -------
        int
            index of parent of child index j
        """
        return (child_index - 1) // 2

    def get_left_child_index(self, parent_index: int) -> int:
        """Returns index of left child of parent index

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        int
            index of left child of parent_index
        """
        return 2 * parent_index + 1

    def get_right_child_index(self, parent_index: int) -> int:
        """Returns index of right child of parent index

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        int
            index of right child of parent_index
        """
        return 2 * parent_index + 2

    def has_left_child(self, parent_index: int) -> bool:
        """Returns true if parent index has a left child

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        bool
            true if parent has a left child
        """
        index_of_left_child = self.get_left_child_index(parent_index)
        if index_of_left_child < len(self):
            return True
        return False

    def has_right_child(self, parent_index: int) -> bool:
        """Returns True if parent index has a right child

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        bool
            true if parent has a right child
        """
        index_of_right_child = self.get_right_child_index(parent_index)
        if index_of_right_child < len(self):
            return True
        return False

    def swap_entries(self, i: int, j: int) -> None:
        """Swaps entries at position i and j

        Parameters
        ----------
        i : int
            position to be swapped
        j : int
            position to be swapped
        """
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def upheap_bubbling(self, j: int) -> None:
        """Performs upheap bubbling via recrusion
        T(n) ~ O(h) where h is height of tree
        and h = log(n)

        Parameters
        ----------
        j : int
            index of item inserted into the queue
        """
        parent_index = self.get_parent_index(j)
        if j > 0 and self._data[parent_index] < self._data[j]:
            self.swap_entries(j, parent_index)
            self.upheap_bubbling(parent_index)

    def push(self, key: Any, value: Any) -> None:
        """Adds item to the priority queue and performs upheap bubbling after insertion

        Parameters
        ----------
        key : Any
            [description]
        value : Any
            [description]
        """
        self._data.append(Item(key, value))
        start_index = len(self) - 1
        self.upheap_bubbling(start_index)

    def downheap_bubbling(self, j: int) -> None:
        """Pergforms downheap bubbling via recursion

        Parameters
        ----------
        j : int
            [description]
        """
        left_child_index = None
        right_child_index = None
        large_child_index = None
        if self.has_left_child(j):
            left_child_index = self.get_left_child_index(j)
            large_child_index = left_child_index
        if self.has_right_child(j):
            right_child_index = self.get_right_child_index(j)
        if left_child_index and right_child_index:
            if self._data[right_child_index] > self._data[left_child_index]:
                large_child_index = right_child_index
        if large_child_index:
            if self._data[large_child_index] > self._data[j]:
                self.swap_entries(j, large_child_index)
                self.downheap_bubbling(large_child_index)

    def pop(self):
        if self.is_empty():
            raise EmptyException("PQ is empty")
        # put minimum item at end and remove it
        n = len(self) - 1
        if self._data[0].value == self._data[n].value:
            item: Item = self._data.pop()
            return (item.key, item.value)
        self.swap_entries(0, n)
        item: Item = self._data.pop()
        self.downheap_bubbling(0)
        return (item.key, item.value)


# - Problem definitions
def p1(k: int) -> str:
    cache = []
    cache.append("1")
    for i in range(1, k):
        val = int(cache[i - 1]) * (i + 1)
        cache.append(f"{val}")
    return ",".join(cache[::-1])


def p2_a(x: list, y: list) -> list:
    z = deepcopy(y)
    z.sort(reverse=True)
    return z[:-1]


def p2_b(x: list, y: list) -> list:
    z = deepcopy(x)
    return z[::-1]


def p2_c(x: list, y: list) -> list:
    z = deepcopy(y)
    w = deepcopy(x)
    answer = list(set(z + w))
    answer.sort()
    return answer


def p2_d(x: list, y: list) -> list:
    z = deepcopy(y)
    w = deepcopy(x)
    return [w, z]


def p3_a(x: set, y: set, z: set) -> set:
    return x | y | z


def p3_b(x: set, y: set, z: set) -> set:
    return x & y & z


def p3_c(x: set, y: set, z: set) -> set:
    return (x | y | z) - (x & y) - (x & z) - (y & z) - (x & y & z)


def p4_a():
    result = np.ones((5, 5))
    result[1:4, 1:4] = 0
    result[2, 2] = 2
    return result


def p4_b(x: np.array):
    pawn_pos = np.where(x == 2)
    knight_pos = np.where(x == 1)
    knight_coordinates = set(
        [Coordinate(xidx=k[0], yidx=k[1]) for k in list(zip(*knight_pos))]
    )
    pawn_pos = Coordinate(pawn_pos[0][0], pawn_pos[1][0])
    indices_that_threaten_pawn = []
    # current position can be attacked by either move right, up twice
    if pawn_pos.move_right().move_up().move_up() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    # checking right ward moves
    if pawn_pos.move_right().move_right().move_up() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    if pawn_pos.move_right().move_down().move_down() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    if pawn_pos.move_right().move_right().move_down() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    # checkinig left ward moves
    if pawn_pos.move_left().move_left().move_up() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    if pawn_pos.move_left().move_up().move_up() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    if pawn_pos.move_left().move_left().move_down() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    if pawn_pos.move_left().move_down().move_down() in knight_coordinates:
        indices_that_threaten_pawn.append(pawn_pos.get_coordinates())
    pawn_pos.reset_coordinates()
    return indices_that_threaten_pawn


def p5_a(x: dict) -> int:
    count = 0
    for v in x.keys():
        if x[v]:
            continue
        count += 1
    return count


def p5_b(x: dict) -> int:
    count = 0
    for v in x.keys():
        if x[v]:
            count += 1
            continue
    return count


def p5_c(x: dict) -> list:
    seen = set()
    # iterate through all the vertices
    for v in x.keys():
        if x[v]:
            # grab the vertex in adjacency list
            for k in x[v]:
                edge = Edge(v, k)
                if edge not in seen:
                    # flip it
                    flipped_edge = Edge(edge.end, edge.start)
                    seen.add(flipped_edge)
                    continue

    return [(edge.start, edge.end) for edge in seen]


def get_adjacency_pairs(x):
    edges = p5_c(x)
    vertices = {}
    for idx, vertex in enumerate(x.keys()):
        v = Vertex(vertex, idx)
        vertices[vertex] = v
    for e in edges:
        start_vertex, end_vertex = e[0], e[1]
        yield (vertices[start_vertex].index, vertices[end_vertex].index)
        yield (vertices[end_vertex].index, vertices[start_vertex].index)


def p5_d(x: dict) -> np.array:
    pairs = dict.fromkeys(get_adjacency_pairs(x), 1)
    vertices = range(len(x.keys()))
    n = len(x.keys())
    arr = np.zeros((n, n))
    for p in permutations(vertices, 2):
        i, j = p
        arr[i, j] = pairs.get((i, j), 0)
    return arr


if __name__ == "__main__":

    print(p1(k=8))
    print("-----------------------------")
    print(p2_a(x=[], y=[1, 3, 5]))
    print(p2_b(x=[2, 4, 6], y=[]))
    print(p2_c(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print(p2_d(x=[1, 3, 5, 7], y=[1, 2, 5, 6]))
    print("------------------------------")
    print(p3_a(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_b(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    print(p3_c(x={1, 3, 5, 7}, y={1, 2, 5, 6}, z={7, 8, 9, 1}))
    # problem 4 under attack
    print("------------------------------")
    print(p4_a())
    print(p4_b(p4_a()))
    print("------------------------------")
    graph = {
        "A": ["D", "E"],
        "B": ["E", "F"],
        "C": ["E"],
        "D": ["A", "E"],
        "E": ["A", "B", "C", "D"],
        "F": ["B"],
        "G": [],
    }
    print(p5_a(graph))
    print(p5_b(graph))
    print(p5_c(graph))
    print(p5_d(graph))
    print("------------------------------")
    pq = PriorityQueue()
    pq.push("apple", 5.0)
    pq.push("kiwi", 7.4)
    pq.push("orange", 5.0)

    while not pq.is_empty():
        print(pq.pop())
