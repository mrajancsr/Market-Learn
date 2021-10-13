from __future__ import annotations, division, print_function

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

GOAL_CONFIG = list(range(3 * 3))


class _Empty(Exception):
    pass


@dataclass
class Solution:
    initial_state: PuzzleState
    goal_state: PuzzleState
    explored: Set[Tuple[int]]
    max_depth: int
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


@dataclass
class Item:
    key: PuzzleState
    value: float
    index: int

    def __lt__(self, other: Item) -> bool:
        return self.value < other.value


class EmptyException(Exception):
    pass


@dataclass
class PriorityQueue:
    """List based implementation of priority queue"""

    _data: List[Item] = field(init=False, default_factory=list)
    _set: Dict[Tuple[int, ...], Item] = field(init=False, default_factory=dict)

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
        self._data[i].index, self._data[j].index = i, j

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
        if j > 0 and self._data[parent_index] > self._data[j]:
            self.swap_entries(j, parent_index)
            self.upheap_bubbling(parent_index)

    def update(self, item: Item, newkey: PuzzleState, newvalue: float) -> None:
        j = item.index
        if not (0 <= j < len(self) and self._data[j] is item):
            raise ValueError("Invalid index")
        item.key = newkey
        item.value = newvalue
        self.heap_bubble(j)

    def heap_bubble(self, j: int) -> None:
        if j > 0 and self._data[j] < self._data[self.get_parent_index(j)]:
            self.upheap_bubbling(j)
        else:
            self.downheap_bubbling(j)

    def push(self, key: PuzzleState, value: float) -> Item:
        """Adds item to the priority queue and performs upheap bubbling after insertion

        Parameters
        ----------
        key : Any
            [description]
        value : Any
            [description]
        """
        n = len(self)
        item = Item(key, value, n)
        self._data.append(item)
        start_index = len(self) - 1
        self.upheap_bubbling(start_index)

        self._set[tuple(key.config)] = item
        return item

    def downheap_bubbling(self, j: int) -> None:
        """Pergforms downheap bubbling via recursion

        Parameters
        ----------
        j : int
            [description]
        """
        left_child_index = None
        right_child_index = None
        small_child_index = None
        if self.has_left_child(j):
            left_child_index = self.get_left_child_index(j)
            small_child_index = left_child_index
        if self.has_right_child(j):
            right_child_index = self.get_right_child_index(j)
        if left_child_index and right_child_index:
            if self._data[right_child_index] < self._data[left_child_index]:
                small_child_index = right_child_index
        if small_child_index:
            if self._data[small_child_index] < self._data[j]:
                self.swap_entries(j, small_child_index)
                self.downheap_bubbling(small_child_index)

    def pop(self):
        if self.is_empty():
            raise EmptyException("PQ is empty")
        n = len(self) - 1
        # put minimum item at end, remove it
        self.swap_entries(0, n)
        item: Item = self._data.pop()
        # perform downheap bubbling starting from root
        self.downheap_bubbling(0)
        del self._set[tuple(item.key.config)]
        return (item.key, item.value)


class Stack:
    """Implementation of stacks using lists

    O(1) time for all operations
    O(n) time for max operation
    Assumes LIFO;
    items on the right is the top, left is bottom
    """

    def __init__(self):
        self._data = []
        self._set = set()

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        yield from self._data

    def __repr__(self):
        return repr(self._data)

    def empty(self) -> bool:
        """returns true if stack is empty"""
        return self._data == []

    def pop(self) -> PuzzleState:
        """removes item from right(top) of stack"""
        if self.empty():
            raise _Empty("Stack is empty")
        item = self._data.pop()
        self._set.remove(tuple(item.config))
        return item

    def push(self, item: PuzzleState) -> None:
        """adds item to right(top) of stack

        :param item: item to be pushed
        :type item: Any
        """
        self._data.append(item)
        self._set.add(tuple(item.config))


class Queue:
    """Implementation of a Queue using a list

    O(1) time for all operations
    Assumes: First In First Out (FIFO)
    """

    # class constant
    _DEFAULT_CAPACITY = 10

    def __init__(self):
        """Default Constructor, needs no parameters"""
        self._data = [None] * Queue._DEFAULT_CAPACITY
        self._size = 0
        self._front = 0
        self._set = set()

    def __len__(self):
        return self._size

    def size(self):
        return len(self)

    def empty(self):
        return self.size() == 0

    def peek(self):
        """Returns, but doesn't remove first element"""
        if self.empty():
            raise IndexError("Queue is Empty")
        return self._data[self._front]

    def dequeue(self):
        """Removes items from the front of queue"""
        if self.empty():
            raise IndexError("Queue is Empty")
        removed_item: PuzzleState = self.peek()
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

    def _resize(self, cap):
        old = self._data
        self._data = [None] * cap
        curr = self._front
        for k in range(self.size()):
            self._data[k] = old[curr]
            curr = (1 + curr) % len(old)
        self._front = 0


class PuzzleState(object):
    """
    The PuzzleState stores a board configuration and implements
    movement instructions to generate valid children.
    """

    def __init__(
        self,
        config,
        n,
        parent=None,
        action="Initial",
        cost=0,
        depth=0,
        blank_index=None,
    ):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n * n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n * n)):
            raise Exception(
                "Config contains invalid/duplicate entries : ", config
            )

        self.n = n
        self.cost = cost
        self.parent = parent
        self.action = action
        self.config = config
        self.children = []
        self.depth = depth

        # Get the index and (row, col) of empty block
        self.blank_index = (
            self.config.index(0) if blank_index is None else blank_index
        )

    def display(self):
        """Display this Puzzle state as a n*n board"""
        for i in range(self.n):
            print(self.config[3 * i : 3 * (i + 1)])

    def move_up(self):
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
            depth=self.depth + 1,
            blank_index=swap_index,
        )

    def move_down(self):
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

    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
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

    def move_right(self):
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

    def expand(self):
        """Generate the child nodes of this node"""

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


def writeOutput(
    solution: Solution,
):
    print(f"path_to_goal: {repr([state.action for state in solution.path])}")
    print(f"cost_of_path: {solution.path[-1].cost}")
    print(f"nodes_expanded: {len(solution.explored) - 1}")
    print(f"search_depth: {solution.path[-1].depth}")
    print(f"max_search_depth: {solution.max_depth}")


def bfs_search(initial_state):
    """BFS search"""
    frontier = Queue()
    frontier.enqueue(initial_state)
    max_depth = 0
    explored = set()
    while not frontier.empty():
        state = frontier.dequeue()
        max_depth = max(max_depth, state.depth)
        explored.add(tuple(state.config))
        if test_goal(state):
            return Solution(initial_state, state, explored, max_depth)
        for neighbor in iter(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if (
                neighbor_config not in frontier._set
                and neighbor_config not in explored
            ):
                frontier.enqueue(neighbor)
    return None


def dfs_search(initial_state):
    """DFS search"""
    frontier = Stack()
    frontier.push(initial_state)
    max_depth = 0
    explored = set()
    while not frontier.empty():
        state: PuzzleState = frontier.pop()
        max_depth = max(max_depth, state.depth)
        explored.add(tuple(state.config))
        if test_goal(state):
            return Solution(initial_state, state, explored, max_depth)
        for neighbor in reversed(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if (
                neighbor_config not in frontier._set
                and neighbor_config not in explored
            ):
                frontier.push(neighbor)
    return None


def A_star_search(initial_state: PuzzleState):
    """A * search"""
    frontier = PriorityQueue()
    total_cost = calculate_total_cost(initial_state)

    frontier.push(initial_state, total_cost)
    explored = set()
    while not frontier.is_empty():
        state, cost = frontier.pop()
        explored.add(tuple(state.config))
        if test_goal(state):
            return Solution(initial_state, state, explored, 0)

        for neighbor in iter(state.expand()):
            neighbor_config = tuple(neighbor.config)
            if (
                neighbor_config not in frontier._set
                and neighbor_config not in explored
            ):
                frontier.push(neighbor, calculate_total_cost(neighbor))
            elif neighbor_config in frontier._set:
                item = frontier._set[neighbor_config]
                frontier.update(item, neighbor, calculate_total_cost(neighbor))
    return None


def calculate_total_cost(state: PuzzleState):
    """calculate the total estimated cost of a state"""
    mhd = sum(
        calculate_manhattan_dist(idx, value, 3)
        for idx, value in enumerate(state.config)
    )
    return state.cost + mhd


def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    if idx == value:
        return 0
    # zero is not a tile
    elif value == 0:
        return 0
    else:
        term1 = divmod(value, n)
        term2 = divmod(idx, n)
        return sum(abs(x - y) for x, y in zip(term1, term2))


def test_goal(puzzle_state: PuzzleState) -> bool:
    """test the state is the goal state or not"""
    return puzzle_state.config == GOAL_CONFIG


# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    begin_state = [1, 2, 5, 3, 4, 0, 6, 7, 8]
    board_size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, board_size)
    start_time = time.time()
    solution = dfs_search(hard_state)
    writeOutput(solution)
    solution = bfs_search(hard_state)
    print("/n")
    writeOutput(solution)
    solution = A_star_search(hard_state)
    writeOutput(solution)
    end_time = time.time()
    print("Program completed in %.3f second(s)" % (end_time - start_time))


if __name__ == "__main__":
    main()
