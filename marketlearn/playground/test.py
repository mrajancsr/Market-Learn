from __future__ import annotations

import datetime

# pyre-strict
from dataclasses import dataclass
from typing import List, Optional, Union
from unittest.mock import Mock

tuesday = datetime.datetime(year=2019, month=1, day=1)
saturday = datetime.datetime(year=2019, month=1, day=5)

datetime = Mock()


def is_weekday():
    today = datetime.datetime.today()
    return 0 <= today.weekday() < 5


datetime.datetime.today.return_value = tuesday
assert is_weekday()

datetime.datetime.today.return_value = saturday
assert not is_weekday()


@dataclass
class TreeNode:
    val: Union[str, int]
    right: Optional[TreeNode] = None
    left: Optional[TreeNode] = None


class Stack:
    def __init__(self):
        self.data = []

    def push(self, item):
        self.data.append(item)

    def pop(self):
        return self.data.pop()

    def empty(self):
        return self.data == []

    def peek(self):
        return self.data[-1]


def inorder(tree: TreeNode) -> List[int]:
    return inorder(tree.left) + [tree.val] + inorder(tree.right) if tree else []


def inorderiterative(root: TreeNode) -> List[int]:
    s = Stack()
    while not s.empty() or root is not None:
        if root:
            s.push(root)
            root = root.left
        else:
            curr = s.peek()
            s.pop()
            yield curr.val
            root = curr.right


def main():
    tree = TreeNode(10)
    left_child = TreeNode(7, left=TreeNode(5), right=TreeNode(8))
    right_child = TreeNode(15, left=TreeNode(12), right=TreeNode(16))
    tree.left = left_child
    tree.right = right_child
    print(inorder(tree))
    print(list(inorderiterative(tree)))


if __name__ == "__main__":
    main()
