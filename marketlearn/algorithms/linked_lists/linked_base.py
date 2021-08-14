"""Abstract Base class for Doubly Linked List"""
# pyre-strict
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


class EmptyException(Exception):
    pass


@dataclass
class Position(metaclass=ABCMeta):
    """Abstract Base Class representing position of an element"""

    @abstractmethod
    # pyre-fixme
    def element(self) -> Any:
        """returns element stored in this position"""
        pass

    @abstractmethod
    def __eq__(self, other: Position) -> bool:
        """returns True if other position represents same location"""
        pass

    def __ne__(self, other: Position) -> bool:
        """returns True if other does not represent the same location

        Parameters
        ----------
        other : Position
            position of single element

        Returns
        -------
        bool
            True if other does not represent same location
        """
        return not (self == other)


@dataclass
class Node:
    """
    Create a node to store v alue and 2 references for LinkedList
    Takes O(1) time

    Parameters
    ----------
    element : Any
        represents element of the node
    nref : optional[Node], default=None
        next referernce of the node
    pref : optional[Node], default=None
        previous reference of the node
    """

    # pyre-fixme
    element: Any
    pref: Optional[Node] = None
    nref: Optional[Node] = None


@dataclass
class DoublyLinkedBase(metaclass=ABCMeta):
    start_node: Node = field(init=False, default=Node(None))
    end_node: Node = field(init=False, default=Node(None))
    size: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.start_node.nref = self.end_node
        self.end_node.pref = self.start_node

    def __len__(self) -> int:
        return self.size

    def is_empty(self) -> bool:
        """Returns True if empty

        Returns
        -------
        bool
            True if empty
        """
        return self.size == 0

    # pyre-fixme
    def _insert_between(self, data: Any, node1: Node, node2: Node) -> Node:
        """Adds data between two nodes

        Parameters
        ----------
        data : Any
            data to be inserted
        node1 : Node
            previous node
        node2 : Node
            next node

        Returns
        -------
        [Node]
            node inserted between two nodes
        """
        # Create a node
        new_node = Node(data, node1, node2)

        # Set new node between two nodes
        node1.nref = new_node
        node2.pref = new_node
        self.size += 1
        return new_node

    # pyre-fixme
    def _delete_node(self, node: Node) -> Any:
        """Deletes node from list and returns the element

        Parameters
        ----------
        node : Node
            node to be deleted

        Returns
        -------
        Any
            data contained within the node
        """
        # save the nodes previous and next reference prior to deletion
        before = node.pref
        after = node.nref
        if before:
            before.nref = after
        if after:
            after.pref = before
        self.size -= 1
        element = node.element

        # deprecate the node
        node.nref = node.pref = node.element = None
        return element

    def _traverse(self) -> None:
        """Traverses a Linked List
        Takes O(n) time
        """
        if self.start_node is None:
            print("list has no elements")
        else:
            n = self.start_node
            while n is not None:
                print(n._element, " ")
                n = n._nref
