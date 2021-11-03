from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, permutations
from typing import Dict, List, Optional, Set, Tuple

ROW = "ABCDEFGHI"
COL = "123456789"

ROWS: Dict[str, Tuple[str, ...]] = {
    "A": ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"),
    "B": ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"),
    "C": ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"),
    "D": ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"),
    "E": ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"),
    "F": ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"),
    "G": ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"),
    "H": ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9"),
    "I": ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9"),
}

COLUMNS: Dict[str, Tuple[str, ...]] = {
    "1": ("A1", "B1", "C1", "D1", "E1", "F1", "G1", "H1", "I1"),
    "2": ("A2", "B2", "C2", "D2", "E2", "F2", "G2", "H2", "I2"),
    "3": ("A3", "B3", "C3", "D3", "E3", "F3", "G3", "H3", "I3"),
    "4": ("A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4", "I4"),
    "5": ("A5", "B5", "C5", "D5", "E5", "F5", "G5", "H5", "I5"),
    "6": ("A6", "B6", "C6", "D6", "E6", "F6", "G6", "H6", "I6"),
    "7": ("A7", "B7", "C7", "D7", "E7", "F7", "G7", "H7", "I7"),
    "8": ("A8", "B8", "C8", "D8", "E8", "F8", "G8", "H8", "I8"),
    "9": ("A9", "B9", "C9", "D9", "E9", "F9", "G9", "H9", "I9"),
}

BOXES: Dict["str", Tuple[str, ...]] = {
    "1": ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"),
    "2": ("A4", "A5", "A6", "B4", "B5", "B6", "C4", "C5", "C6"),
    "3": ("A7", "A8", "A9", "B7", "B8", "B9", "C7", "C8", "C9"),
    "4": ("D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3"),
    "5": ("D4", "D5", "D6", "E4", "E5", "E6", "F4", "F5", "F6"),
    "6": ("D7", "D8", "D9", "E7", "E8", "E9", "F7", "F8", "F9"),
    "7": ("G1", "G2", "G3", "H1", "H2", "H3", "I1", "I2", "I3"),
    "8": ("G4", "G5", "G6", "H4", "H5", "H6", "I4", "I5", "I6"),
    "9": ("G7", "G8", "G9", "H7", "H8", "H9", "I7", "I8", "I9"),
}


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

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return self._size

    def size(self):
        return len(self)

    def empty(self):
        return self.size() == 0

    def dequeue(self):
        """Removes items from the front of queue"""
        if self.empty():
            raise IndexError("Queue is Empty")
        removed_item = self._data[self._front]
        # reclaim for garbage collection
        self._data[self._front] = None
        # get available space and decrement size by 1
        self._front = (1 + self._front) % len(self._data)
        self._size -= 1
        return removed_item

    def enqueue(self, item):
        # if capcacity is reached, double capacity
        if self.size() == len(self._data):
            self._resize(2 * len(self._data))
        avail = (self._front + self.size()) % len(self._data)
        self._data[avail] = item
        self._size += 1

    def _resize(self, cap):
        old = self._data
        self._data = [None] * cap
        curr = self._front
        for k in range(self.size()):
            self._data[k] = old[curr]
            curr = (1 + curr) % len(old)
        self._front = 0


@dataclass
class Constraint:
    """Binary Constraint class for sodoku solver"""

    first: str
    second: str
    variables: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.variables = [self.first, self.second]

    def __hash__(self):
        return hash((self.first, self.second))

    def __eq__(self, other: Constraint):
        return (
            type(other) is type(self) and other.endpoint() == self.endpoint()
        )

    def __repr__(self) -> str:
        return f"Constraint({self.first} != {self.second})"

    def endpoint(self) -> Tuple[str, str]:
        return (self.first, self.second)

    def flip(self) -> Tuple[str, str]:
        return (self.second, self.first)

    def satisfied(self, assignment: Dict[str, int]) -> bool:
        if self.first not in assignment or self.second not in assignment:
            return True
        # check that number assigned to first variable
        # is not the same as number assigned to constrained variable
        return assignment[self.first] != assignment[self.second]


def split(string: str):
    head = string.split("123456789")
    tail = string[len(head) :]
    return head, int(tail)


@dataclass
class CSP:
    """Constraint Satisfaction Problem Class"""

    variables: List[str]
    domains: Dict[str, Set[int]]
    constraints: Dict[str, Dict[str, Set[Constraint]]] = field(
        init=False, default_factory=dict
    )
    _set: Dict[str, Set[Tuple[str, str]]] = field(
        init=False, default_factory=dict
    )
    _removed_domains: Dict[str, Dict[str, int]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self.constraints = {k: defaultdict(set) for k in self.variables}
        self._set = {k: set() for k in self.variables}
        self._removed_domains = defaultdict(dict)
        assert self.variables == [
            *self.domains
        ], "Look Up Error, every variable must have a domain"

    def add_constraint(self, constraint: Constraint) -> None:
        """Adds a constraint to each variable in CSP

        Parameters
        ----------
        constraint : Constraint
            [description]

        Raises
        ------
        LookupError
            [description]
        """
        first_variable, second_variable = constraint.variables
        if (
            first_variable not in self.constraints
            or second_variable not in self.constraints
        ):
            raise LookupError(
                " The variable in constraint is not a variable in CSP"
            )

        self.constraints[first_variable][second_variable].add(constraint)
        self._set[first_variable].add((constraint.first, constraint.second))

    def consistent(self, variable: str, assignment: Dict[str, int]) -> bool:
        """Check if every constraint with respect to variable satisfies assignment

        Parameters
        ----------
        variable : str
            [description]
        assignment : Dict[str, int]
            [description]

        Returns
        -------
        bool
            [description]
        """
        for constraint in chain.from_iterable(
            self.constraints[variable].values()
        ):
            if not constraint.satisfied(assignment):
                return False
        return True

    def _remaining_domain(
        self, unassigned_variable: str, assignment: Dict[str, int]
    ):
        # get constraints for unassigned variable
        impossible_values = {
            assignment[k]
            for k in chain.from_iterable(self._set[unassigned_variable])
            if k != unassigned_variable and k in assignment
        }
        return self.domains[unassigned_variable] - impossible_values

    def select_unassigned_variable(
        self,
        unassigned: List[str],
        assignment: Dict[str, int],
    ) -> Dict[str, List[int]]:
        """Selects variable based on MRV heuristic

        Parameters
        ----------
        unassigned : List[str]
            unassigned variables
        assignment : Dict[str, int]
            assigned variables

        Returns
        -------
        Dict[str, List[int]]
            [description]
        """
        remaining_domain = {}
        for variable in unassigned:
            remaining_domain[variable] = self._remaining_domain(
                variable, assignment
            )
        min_val = min(map(lambda x: len(x), remaining_domain.values()))
        return {k: v for k, v in remaining_domain.items() if len(v) == min_val}


def backtracking_search(
    csp: CSP, assignment: Dict[str, int] = {}
) -> Optional[Dict[str, int]]:
    # if all the variables are assigned, return assignment
    if len(assignment) == len(csp.variables):
        return assignment
    unassigned = [v for v in csp.variables if v not in assignment]
    # first step: select variable based on minimum remaining value
    mrv = csp.select_unassigned_variable(unassigned, assignment)
    unassigned_variable: str = list(mrv)[0]
    for value in mrv[unassigned_variable]:
        # now variable is assigned
        assignment[unassigned_variable] = value
        if csp.consistent(unassigned_variable, assignment):
            for neighbor in csp.constraints[unassigned_variable].keys():
                _ = revise_domain(
                    csp, neighbor, unassigned_variable, assignment
                )
            result = backtracking_search(csp, assignment)
            if result is not None:
                return result
            del assignment[unassigned_variable]
            # restore previous domain removed
            restore_previous_domain(csp, unassigned_variable)

    return None


def restore_previous_domain(csp: CSP, unassigned_variable: str) -> None:
    """Restores the domain of unassigned variable's neighbor

    Parameters
    ----------
    csp : CSP
        [description]
    unassigned_variable : str
        [description]
    value : int
        [description]
    """
    for neighbor in csp.constraints[unassigned_variable].keys():
        if neighbor in csp._removed_domains[unassigned_variable]:
            removed_item = csp._removed_domains[unassigned_variable][neighbor]
            csp.domains[neighbor].add(removed_item)


def make_network_arc_consistent(csp: CSP, assignment) -> bool:
    """Implementation of the AC-3 algorithm

    Parameters
    ----------
    csp : CSP
        binary CSP constraint
    assignment : [type]
        current assignment of variables in CSP

    Returns
    -------
    bool
        [description]
    """
    queue = set()
    for arcs in csp._set.values():
        queue.update(arcs)
    while len(queue) != 0:
        vi, vj = queue.pop()
        if revise_domain(csp, vi, vj, assignment):
            if len(csp.domains[vi]) == 0:
                return False
            for vk in csp.constraints[vi].keys() - {vj}:
                queue.add((vk, vi))
    return True


def revise_domain(
    csp: CSP,
    first_variable: str,
    second_variable: str,
    assignment: Dict[str, int],
) -> bool:
    """Returns True if we revise domain of first_variable

    Parameters
    ----------
    csp : CSP
        constraint satisfaction problem
    first_variable : str
        the first variable in csp
    second_variable : str
        the second variable in csp

    Returns
    -------
    bool
        True if domain of first_variable is reduced
    """
    revised = False
    if first_variable in assignment and second_variable in assignment:
        pass
    elif first_variable in assignment:
        pass
    elif second_variable in assignment:
        value = assignment[second_variable]
        if value in csp.domains[first_variable]:
            csp.domains[first_variable].remove(value)
            csp._removed_domains[second_variable][first_variable] = value
            revised = True
    else:
        local_assignment = assignment.copy()
        # both the variables are unassigned
        constraint: Constraint = next(
            iter(csp.constraints[first_variable][second_variable])
        )
        first_variable_domain = csp.domains[first_variable].copy()
        for x in first_variable_domain:
            not_satisfied_count = 0
            local_assignment[first_variable] = x
            for y in csp.domains[second_variable]:
                local_assignment[second_variable] = y
                if not constraint.satisfied(local_assignment):
                    not_satisfied_count += 1
            if not_satisfied_count == len(csp.domains[second_variable]):
                csp.domains[first_variable].remove(x)
                csp._removed_domains[second_variable][first_variable] = x
                revised = True

    return revised


def print_board(board):
    """Helper function to print board in a square."""
    print("-----------------")
    for i in ROW:
        row = ""
        for j in COL:
            row += str(board[i + j]) + " "
        print(row)


def board_to_string(board):
    """Helper function to convert board dictionary to string for writing."""
    ordered_vals = []
    for r in ROW:
        for c in COL:
            ordered_vals.append(str(board[r + c]))
    return "".join(ordered_vals)


def backtracking(board):
    """Takes a board and returns solved board."""
    # TODO: implement this
    solved_board = board
    return solved_board


def convert_to_board(string: str):
    board = {
        ROW[r] + COL[c]: int(string[9 * r + c])
        for r in range(9)
        for c in range(9)
    }
    return board


def main(board_num: int):
    # Running sudoku solver with one board $python3 sudoku.py <input_string>.
    # Parse boards to dict representation, scanning board L to R, Up to Down
    import os

    print(os.getcwd())
    path = os.getcwd()
    path_to_file = os.path.join(path, "starter", "sudokus_start.txt")
    with open(path_to_file, "r") as f:
        boards = f.read().splitlines()

    all_boards = []
    for b in boards:
        all_boards.append(convert_to_board(b))

    for board in all_boards[:board_num]:
        pass

    variables: List[str] = list(board.keys())
    # get the assignments
    assignment: Dict[str, int] = {k: v for k, v in board.items() if v != 0}
    domains: Dict[str, Set[int]] = {}
    for k in variables:
        if k not in assignment:
            domains[k] = set(range(1, 10))
        else:
            domains[k] = set([assignment[k]])

    csp: CSP = CSP(variables, domains)

    # add top row constraint
    for row in ROWS:
        for c in permutations(ROWS[row], 2):
            csp.add_constraint(Constraint(*c))

    # add column constraints
    for col in COLUMNS:
        for c in permutations(COLUMNS[col], 2):
            csp.add_constraint(Constraint(*c))

    # add box constraints
    for box in BOXES:
        for c in permutations(BOXES[box], 2):
            csp.add_constraint(Constraint(*c))

    # pre-process step
    make_network_arc_consistent(csp, assignment)
    print("\n")
    print("board provided", "\n")
    print_board(board)
    print("\n")
    print("Solution", "\n")
    print_board(backtracking_search(csp, assignment))


if __name__ == "__main__":
    main(402)
