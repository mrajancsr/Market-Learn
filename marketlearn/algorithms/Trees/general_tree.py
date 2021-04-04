"""Module Implements General Trees using linked lists

Author: Rajan Subramanian
Date: -
"""
from __future__ import annotations
from marketlearn.algorithms.trees import tree_base as tb
from typing import Any, Iterator, Union


class GeneralTree(tb._GeneralTreeBase):
    """Class representing general tree structure using linked representation"""

    class _Node:

        __slots__ = "_element", "_parent", "_left", "_right"

        def __init__(self, element, parent=None, children=None):
            self._element = element
            self._parent = parent
            self._children = children

    class Position(tb.Position):
        """Abstraction representing location of single element"""

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            """return element stored at position """
            return self._node._element

        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node

    def _make_position(self, node: _Node) -> Union[Position, None]:
        """Return Position's instance for a given node"""
        return self.Position(self, node) if node is not None else None

    def _validate(self, p: Position) -> _Node:
        """return position's node or raise appropriate error if invalid

        Parameters
        ----------
        p : Position
            represents the position of interest

        Returns
        -------
        _Node
            position's node object

        Raises
        ------
        TypeError
            if p is not a proper Position
        TypeError
            if p does not belong to same container
        ValueError
            if p's parent is the current node
        """
        if not isinstance(p, self.Position):
            raise TypeError("p must be proper Position type")
        if p._container is not self:
            raise TypeError("p does not belong to this container")

        # convention for deprecated nodes
        if p._node._parent is p._node:
            raise ValueError("p is no longer valid")
        return p._node

    # general tree constructor
    def __init__(self):
        """Creates a initially empty general tree"""
        self._root = None
        self._size = 0

    def __len__(self):
        """returns total number of nodes in a tree"""
        return self._size

    def root(self) -> Position:
        """return root position of tree, return None if tree is empty"""
        return self._make_position(self._root)

    def parent(self, p: Position) -> Union[Position, None]:
        """return position representing p's parent (or None if p is root)

        Parameters
        ----------
        p : Position
            position who's parent we are interested in

        Returns
        -------
        Union[Position, None]
            position representing p's parent or None if p is root
        """
        node = self._validate(p)
        return self._make_position(node._parent)

    def num_children(self, p: Position) -> int:
        """returns count of total children of position p

        Parameters
        ----------
        p : Position
            represents the parent position

        Returns
        -------
        int
            count of total children of p
        """
        pass


gt = GeneralTree()
