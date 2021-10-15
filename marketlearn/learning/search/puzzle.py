from __future__ import annotations, division, print_function

import math
import resource
import sys
import time
from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterator, List, Optional, Set, Tuple

from marketlearn.algorithms.linear_collections import CircularQueue, Stack
from marketlearn.algorithms.priority_queues import HeapPriorityQueue

# - NPuzzle Game from Artificial Intelligence, a modern perspective -
# This module is used to reproduce informed and uninformed search strategies
# on NPuzzle Game.  The goal of this game is that given any initial board
# configuration, we need to determine the total number of movements required
# reach the final board state, given by range(9)

# Graph Search method was used in the implementation since states are repeated
# Module contains implementation of Breadth-First-Search, Depth-First-Search
# and A*-Search

GOAL_CONFIG = list(range(3 * 3))


class _Empty(Exception):
    pass


# utility class to store the solution of search
@dataclass
class Solution:
    initial_state: PuzzleState
    goal_state: PuzzleState
    explored: Set[Tuple[int]]
    max_depth: int
    elapsed_time: float
    max_ram: float
    path: List[PuzzleState] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.path = self.construct_path()

    def construct_path(self):
        path = []
        if tuple(self.goal_state.config) in self.explored:
            # build a path from goal_state to initial_state
            path.append(self.goal_state)
            curr = self.goal_state
            while curr is not self.initial_state:
                parent = curr.parent
                path.append(parent)
                curr = parent
        return path[::-1][1:]


# Utility classes that provide O(1) look up during search
@dataclass
class AdaptablePriorityQueue(HeapPriorityQueue):
    """Allows for updates in priority if priority changes"""

    pass


@dataclass
class sQueue(CircularQueue):
    pass


@dataclass
class sStack(Stack):
    pass


@dataclass
class PuzzleState:
    """Generates Board Configuration of NPuzzleGame"""

    config: List[int]
    n: int
    parent: Optional[PuzzleState] = None
    children: List[PuzzleState] = field(init=False, default_factory=list)
    action: str = "initial"
    cost: int = 0
    depth: int = 0
    blank_index: int = -1

    def __post_init__(self) -> None:
        assert self.n * self.n != len(self.config) or self.n < 2
        assert set(self.config) != set(range(self.n * self.n))
        self.blank_index = (
            self.config.index(0)
            if self.blank_index == -1
            else self.blank_index
        )

    def display(self):
        """Display this Puzzle state as a n*n board"""
        for i in range(self.n):
            print(self.config[3 * i : 3 * (i + 1)])

    def move_up(self) -> Optional[PuzzleState]:
        """
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        n = self.n
        # if top row contians blank, do nothing
        if 0 in self.config[:n]:
            return None

        config = self.config[:]
        swap_index = self.blank_index - n
        config[self.blank_index], config[swap_index] = (
            config[swap_index],
            config[self.blank_index],
        )

        return PuzzleState(
            config,
            n=n,
            parent=self,
            action="Up",
            cost=1 + self.cost,
            depth=1 + self.depth,
            blank_index=swap_index,
        )

    def move_down(self) -> Optional[PuzzleState]:
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        # if bottom row contians blank, do nothing
        n = self.n
        if 0 in self.config[-n:]:
            return None

        config = self.config[:]
        swap_index = self.blank_index + n
        config[self.blank_index], config[swap_index] = (
            config[swap_index],
            config[self.blank_index],
        )

        return PuzzleState(
            config,
            n=n,
            parent=self,
            action="Down",
            cost=1 + self.cost,
            depth=1 + self.depth,
            blank_index=swap_index,
        )

    def move_left(self) -> Optional[PuzzleState]:
        """Moves a tile one column to the left

        Returns
        -------
        Optional[PuzzleState]
            None if move to left is not possible
            otherwise generates the new PuzzleState
        """
        # if top row contians blank, do nothing
        n = self.n
        first_col_slice = slice(0, n * n, n)
        if 0 in self.config[first_col_slice]:
            return None

        config = self.config[:]
        swap_index = self.blank_index - 1
        config[self.blank_index], config[swap_index] = (
            config[swap_index],
            config[self.blank_index],
        )

        return PuzzleState(
            config,
            n=n,
            parent=self,
            action="Left",
            cost=1 + self.cost,
            depth=1 + self.depth,
            blank_index=swap_index,
        )

    def move_right(self) -> Optional[PuzzleState]:
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        n = self.n
        last_col_slice = slice(n - 1, n * n, n)
        if 0 in self.config[last_col_slice]:
            return None

        config = self.config[:]
        swap_index = self.blank_index + 1
        config[self.blank_index], config[swap_index] = (
            config[swap_index],
            config[self.blank_index],
        )

        return PuzzleState(
            config,
            n=n,
            parent=self,
            action="Right",
            cost=1 + self.cost,
            depth=1 + self.depth,
            blank_index=swap_index,
        )

    def expand(self) -> Iterator[PuzzleState]:
        """Generate iteration of children of current node

        Yields
        -------
        Iterator[PuzzleState]
            iterator of children of current node
        """

        # Node has already been expanded
        if len(self.children) != 0:
            yield from self.children

        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right(),
        ]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        yield from self.children


def calculate_total_cost(state: PuzzleState) -> float:
    """Calculates the total estimates cost of a state

    f(n) = g(n) + h(n)
    g(n) is path cost to node n
    h(n) is manhattan distance heuristic from n to goal

    Parameters
    ----------
    state : PuzzleState
        current board state

    Returns
    -------
    [type]
        [description]
    """
    mhd = sum(
        manhattan_distance(idx, value, 3)
        for idx, value in enumerate(state.config)
    )
    return state.cost + mhd


def manhattan_distance(idx: int, value: int, n: int):
    """Calculates the manhattan distance of a tile

    Parameters
    ----------
    idx : int
        [description]
    value : int
        [description]
    n : int
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if idx == value or value == 0:
        return 0
    else:
        return sum(
            abs(x - y) for x, y in zip(divmod(value, n), divmod(idx, n))
        )


def test_goal(puzzle_state: PuzzleState) -> bool:
    """test the state is the goal state or not"""
    return puzzle_state.config == GOAL_CONFIG


def bfs_search(initial_state):
    """BFS search"""
    start = perf_counter()
    frontier = Queue()
    frontier.enqueue(initial_state)
    max_depth = 0
    max_ram = 0
    explored = set()
    while not frontier.empty():
        state = frontier.dequeue()
        explored.add(tuple(state.config))
        if test_goal(state):
            end = perf_counter()
            elapsed_time = end - start
            writeOutput(
                Solution(
                    initial_state,
                    state,
                    explored,
                    max_depth,
                    elapsed_time,
                    max_ram,
                )
            )
            return
        for neighbor in iter(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if (
                neighbor_config not in frontier._set
                and neighbor_config not in explored
            ):
                frontier.enqueue(neighbor)
                max_depth = max(max_depth, neighbor.cost)
                ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                max_ram = max(max_ram, ram)
    return None


def dfs_search(initial_state):
    """DFS search"""
    start = perf_counter()
    frontier = Stack()
    frontier.push(initial_state)
    max_depth = 0
    max_ram = 0
    explored = set()
    while not frontier.empty():
        state: PuzzleState = frontier.pop()
        max_depth = max(max_depth, state.cost)
        explored.add(tuple(state.config))
        if test_goal(state):
            end = perf_counter()
            elapsed_time = end - start
            writeOutput(
                Solution(
                    initial_state,
                    state,
                    explored,
                    max_depth,
                    elapsed_time,
                    max_ram,
                )
            )
            return
        for neighbor in reversed(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if (
                neighbor_config not in frontier._set
                and neighbor_config not in explored
            ):
                frontier.push(neighbor)
                ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                max_ram = max(max_ram, ram)
    return None


def A_star_search(initial_state: PuzzleState):
    """A * search"""
    start = perf_counter()
    frontier = PriorityQueue()
    total_cost = calculate_total_cost(initial_state)
    max_depth = 0
    max_ram = 0
    frontier.push(initial_state, total_cost)
    explored = set()
    while not frontier.is_empty():
        state, _ = frontier.pop()
        max_depth = max(max_depth, state.cost)
        explored.add(tuple(state.config))
        if test_goal(state):
            end = perf_counter()
            elapsed_time = end - start
            writeOutput(
                Solution(
                    initial_state,
                    state,
                    explored,
                    max_depth,
                    elapsed_time,
                    max_ram,
                )
            )
            return

        for neighbor in iter(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if (
                neighbor_config not in frontier._set
                and neighbor_config not in explored
            ):
                frontier.push(neighbor, calculate_total_cost(neighbor))
                ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                max_ram = max(max_ram, ram)
            elif neighbor_config in frontier._set:
                item = frontier._set[neighbor_config]
                frontier.update(item, neighbor, calculate_total_cost(neighbor))
    return None
