import sys

"""
Each sudoku board is represented as a dictionary with string keys and
int values.
e.g. my_board['A1'] = 8
"""
ROW = "ABCDEFGHI"
COL = "123456789"

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set


@dataclass
class Constraint:
    """Constraint Class
    Each constraint is on the variable it constraints"""

    variables: List[str]

    def satisfied(self, assignment: Dict[str, int]) -> bool:
        pass


def split(string: str):
    head = string.split("123456789")
    tail = string[len(head) :]
    return head, int(tail)


@dataclass
class CSP:
    """Constraint Satisfaction Problem Class"""

    variables: List[str]
    domains: Dict[str, List[int]]
    constraints: Dict[str, List[Constraint]] = field(init=False, default={})

    def __post_init__(self) -> None:
        self.constraints = dict.fromkeys(self.variables, [])
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
                self.constraints[variable].append(constraint)

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
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True

    def _get_box_assignment(
        self,
        letter: str,
        num: int,
        assigned: List[str],
        assignment: Dict[str, int],
    ) -> Iterator[int]:
        for v in assigned:
            curr_letter = v[0]
            curr_num = v[1]
            if letter in ["A", "B", "C"] and curr_letter <= "C":
                if all(item in [1, 2, 3] for item in [num, curr_num]):
                    yield assignment[v]
                elif all(item in [4, 5, 6] for item in [num, curr_num]):
                    yield assignment[v]
                elif all(item in [7, 8, 9] for item in [num, curr_num]):
                    yield assignment[v]
            elif letter in ["D", "E", "F"] and curr_letter <= "F":
                if all(item in [1, 2, 3] for item in [num, curr_num]):
                    yield assignment[v]
                elif all(item in [4, 5, 6] for item in [num, curr_num]):
                    yield assignment[v]
                elif all(item in [7, 8, 9] for item in [num, curr_num]):
                    yield v[1]
            elif letter in ["G", "H", "I"] and curr_letter <= "I":
                if all(item in [1, 2, 3] for item in [num, curr_num]):
                    yield assignment[v]
                elif all(item in [4, 5, 6] for item in [num, curr_num]):
                    yield assignment[v]
                elif all(item in [7, 8, 9] for item in [num, curr_num]):
                    yield assignment[v]
            else:
                raise LookupError(
                    "Variable Not found in csp list of variables"
                )

    def _remaining_value(
        self, variable: str, assigned: List[str], assignment: Dict[str, int]
    ):
        letter, num = split(variable)
        impossible_values = set()
        # get row/column values for assigned variables
        impossible_values.update(
            (assignment[v] for v in assigned if v[0] == letter or v[1] == num)
        )
        # get values assigned to box (9 boxes)
        impossible_values.update(
            (self._get_box_assignment(letter, num, assigned, assignment))
        )
        return list(set(self.domains[variable]) - impossible_values)

    def remaining_value(
        self,
        assigned: List[str],
        unassigned: List[str],
        assignment: Dict[str, int],
    ) -> Dict[str, List[int]]:
        result = {}
        for variable in unassigned:
            result[variable] = self._remaining_value(
                variable, assigned, assignment
            )
        return result

    def minimum_remaining_value(
        self,
        assignment: Dict[str, int],
    ) -> Dict[str, List[int]]:
        """Returns the variables that have the smallest values remaining"""
        unassigned = [v for v in self.variables if assignment[v] == 0]
        assigned = list(assignment.keys() - set(unassigned))
        result = self.remaining_value(assigned, unassigned, assignment)
        min_val = min(map(lambda x: len(x), result.values()))
        return {k: v for k, v in result.items() if v == min_val}

    def backtracking_search(
        self, assignment: Dict[str, int] = {}
    ) -> Optional[Dict[str, int]]:
        # if all the variables are assigned, return assignment
        if len(assignment) == len(self.variables):
            return assignment

        # first step: select variable based on mrv as the heuristic
        first: str = ""
        mrv = self.minimum_remaining_value(assignment)
        first: str = list(mrv)[0]
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


if __name__ == "__main__":
    if len(sys.argv) > 1:

        # Running sudoku solver with one board $python3 sudoku.py <input_string>.
        print(sys.argv[1])
        # Parse boards to dict representation, scanning board L to R, Up to Down
        board = {
            ROW[r] + COL[c]: int(sys.argv[1][9 * r + c])
            for r in range(9)
            for c in range(9)
        }

        print_board(board)
        print(board.keys())
