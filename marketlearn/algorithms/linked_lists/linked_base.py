"""Abstract Base class for Doubly Linked List"""

from __future__ import annotations
from typing import Any
from abc import ABCMeta, abstractmethod


class Position(metaclass=ABCMeta):
    """Abstract Base Class representing position of an element"""

    @abstractmethod
    def element(self):
        """returns element stored in this position"""
        pass

    @abstractmethod
    def __eq__(self, other):
        """returns True if other position represents same location"""
        pass

    def __ne__(self, other):
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


class _DoublyLinkedBase(metaclass=ABCMeta):
    class _Node:
        def __init__(
            self, data, previous_ref: _Node = None, next_ref: _Node = None
        ):
            self._element = data
            self._pref = previous_ref
            self._nref = next_ref

    def __init__(self):
        self._start_node = self._Node(None)
        self._end_node = self._Node(None)
        self._start_node._nref = self._end_node  # creating circular reference
        self._end_node._pref = self._start_node
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self) -> bool:
        """Returns True if empty

        Returns
        -------
        bool
            True if empty
        """
        return self._size == 0

    def _insert_between(self, data: Any, node1: _Node, node2: _Node) -> _Node:
        """Adds data between two nodes

        Parameters
        ----------
        data : Any
            data to be inserted
        node1 : _Node
            previous node
        node2 : _Node
            next node

        Returns
        -------
        [_Node]
            node inserted between two nodes
        """
        # Create a node
        new_node = self._Node(data, node1, node2)

        # Set new node between two nodes
        node1._nref = new_node
        node2._pref = new_node
        self._size += 1
        return new_node

    def _delete_node(self, node: _Node) -> Any:
        """Deletes node from list and returns the element

        Parameters
        ----------
        node : _Node
            node to be deleted

        Returns
        -------
        Any
            data contained within the node
        """
        # save the nodes previous and next reference prior to deletion
        before = node._pref
        after = node._nref
        before._nref = after
        after._pref = before
        self._size -= 1
        element = node._element

        # deprecate the node
        node._nref = node._pref = node._element = None
        return element

    def _traverse(self):
        """Traverses a Linked List
        Takes O(n) time
        """
        if self._start_node is None:
            print("list has no elements")
        else:
            n = self._start_node
            while n is not None:
                print(n._element, " ")
                n = n._nref
