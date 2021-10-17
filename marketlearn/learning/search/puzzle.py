from __future__ import annotations, division, print_function

import os
import resource
from dataclasses import dataclass, field
from time import perf_counter
from typing import List, Optional, Set, Tuple

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
OUTPUT_FILE = "output.txt"
PATH_TO_FILE = (
    "/Users/raj/Documents/QuantResearch/Home/market-learn/marketlearn"
)
PATH_TO_FILE = os.path.join(PATH_TO_FILE, "learning", "search", OUTPUT_FILE)


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
    _set: Set[Tuple[int]] = field(init=False, default_factory=set)

    def __post_init__(self):
        super().__init__()

    def dequeue(self):
        """Removes items from the front of queue"""
        if self.empty():
            raise IndexError("Queue is Empty")
        removed_item: PuzzleState = self._data[self._front]
        # reclaim for garbage collection
        self._data[self._front] = None
        # get available space and decrement size by 1
        self._front = (1 + self._front) % len(self._data)
        self._size -= 1
        self._set.remove(tuple(removed_item.config))
        return removed_item

    def enqueue(self, item: PuzzleState):
        # if capcacity is reached, double capacity
        if self.size() == len(self._data):
            self._resize(2 * len(self._data))
        avail = (self._front + self.size()) % len(self._data)
        self._data[avail] = item
        self._set.add(tuple(item.config))
        self._size += 1


@dataclass
class sStack(Stack):
    _set: Set[Tuple[int]] = field(init=False, default_factory=set)

    def pop(self) -> PuzzleState:
        """removes item from right(top) of stack"""
        if self.empty():
            raise _Empty("Stack is empty")
        item = self._data.pop()
        self._set.remove(tuple(item.config))
        return item

    def push(self, item: PuzzleState) -> None:
        """Adds an item to right(top) of stack

        Parameters
        ----------
        item : PuzzleState
            [description]
        """
        self._data.append(item)
        self._set.add(tuple(item.config))


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

    def __repr__(self):
        return self.action

    def __post_init__(self) -> None:
        assert self.n * self.n == len(self.config) and self.n >= 3
        assert set(self.config) == set(range(self.n * self.n))
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
        """Moves blank tile one row down

        Returns
        -------
        Optional[PuzzleState]
            new state after the move or None if at bottom
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

    def expand(self) -> List[PuzzleState]:
        """Generate iteration of children of current node

        Yields
        -------
        List[PuzzleState]
            list of children of current node
        """

        # Node has already been expanded
        if len(self.children) != 0:
            return self.children

        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right(),
        ]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children


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


def manhattan_distance(idx: int, value: int, n: int) -> int:
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


def bfs_search(initial_state) -> Optional[Solution]:
    """BFS search"""
    start = perf_counter()
    frontier = sQueue()
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
            return Solution(
                initial_state,
                state,
                explored,
                max_depth,
                elapsed_time,
                max_ram,
            )
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


def dfs_search(initial_state) -> Optional[Solution]:
    """DFS search"""
    start = perf_counter()
    frontier = sStack()
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
            return Solution(
                initial_state,
                state,
                explored,
                max_depth,
                elapsed_time,
                max_ram,
            )
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


def A_star_search(initial_state: PuzzleState) -> Optional[Solution]:
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
            return Solution(
                initial_state,
                state,
                explored,
                max_depth,
                elapsed_time,
                max_ram,
            )

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


def writeOutput(
    solution: Solution,
):

    with open(PATH_TO_FILE, "w") as f:
        f.write(
            f"path_to_goal: {repr([state.action for state in solution.path])}\n"
        )
        f.write(f"cost_of_path: {solution.path[-1].cost}\n")
        f.write(f"nodes_expanded: {len(solution.explored) - 1}\n")
        f.write(f"search_depth: {solution.path[-1].cost}\n")
        f.write(f"max_search_depth: {solution.max_depth}\n")
        f.write(f"running_time: {solution.elapsed_time}\n")
        f.write(f"max_ram_usage: {solution.max_ram / 1000000.}\n")


# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    state = [6, 1, 8, 4, 0, 2, 7, 3, 5]
    initial_state = PuzzleState(state, 3)
    writeOutput(bfs_search(initial_state))


if __name__ == "__main__":
    main()
