from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain, combinations, permutations
from typing import Dict, Iterator, List, Optional, Set, Tuple

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

    def __post_init__(self) -> None:
        self.constraints = {k: defaultdict(set) for k in variables}
        self._set = {k: set() for k in self.variables}
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
        for variable in constraint.variables:
            if variable not in self.constraints:
                raise LookupError(
                    "The variable in constraint is not a variable in CSP"
                )
            else:
                if (constraint.second, constraint.first) in self._set[
                    variable
                ]:
                    continue
                v = (
                    constraint.first
                    if constraint.first != variable
                    else constraint.second
                )
                self.constraints[variable][v].add(constraint)
                self._set[variable].add((constraint.first, constraint.second))

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

    def revise(self, first_variable: str, second_variable: str) -> bool:
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
        assignment: Dict[str, int] = {}
        constraint: Constraint = next(
            iter(self.constraints[first_variable][second_variable])
        )
        revised = False
        for x in self.domains[first_variable]:
            assignment[first_variable] = x
            for y in self.domains[second_variable]:
                assignment[second_variable] = y
                if not constraint.satisfied(assignment):
                    self.domains[first_variable].remove(x)
                    revised = True

        return revised

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

    def value_selection(
        self,
        selected_variable: str,
        unassigned: List[str],
        assignment: Dict[str, int],
    ) -> str:
        """Selects a value based on least constraining value

        Parameters
        ----------
        selected_variable : str
            [description]
        unassigned : List[str]
            [description]
        assignment : Dict[str, int]
            [description]

        Returns
        -------
        str
            [description]
        """
        pass

    def backtracking_search(
        self, assignment: Dict[str, int] = {}
    ) -> Optional[Dict[str, int]]:
        # if all the variables are assigned, return assignment
        if len(assignment) == len(self.variables):
            return assignment

        unassigned = [v for v in self.variables if v not in assignment]
        # first step: select variable based on minimum remaining value
        mrv = self.select_unassigned_variable(unassigned, assignment)
        first: str = random.sample(list(mrv), 1)[0]
        for value in mrv[first]:
            local_assignment = assignment.copy()
            local_assignment[first] = value
            if self.consistent(first, local_assignment):
                return self.backtracking_search(local_assignment)
        return None


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


if __name__ == "__main__":
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
    result = []
    failure = []
    for board in all_boards[:3]:
        variables: List[str] = list(board.keys())
        domains: Dict[str, Set[int]] = {
            k: set(range(1, 10)) for k in variables
        }

        csp: CSP = CSP(variables, domains)

        # add top row constraint
        for row in ROWS:
            for c in combinations(ROWS[row], 2):
                csp.add_constraint(Constraint(*c))

        # add column constraints
        for col in COLUMNS:
            for c in combinations(COLUMNS[col], 2):
                csp.add_constraint(Constraint(*c))

        # add box constraints
        for box in BOXES:
            for c in combinations(BOXES[box], 2):
                csp.add_constraint(Constraint(*c))
        # get the assignments
        assignment: Dict[str, int] = {k: v for k, v in board.items() if v != 0}
        try:
            result.append(csp.backtracking_search(assignment))
        except Exception:
            failure.append(board)

    for b in result:
        print_board(b)
