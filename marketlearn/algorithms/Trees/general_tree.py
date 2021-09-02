"""Module Implements General Trees using linked lists

Author: Rajan Subramanian
Date: -
"""
from __future__ import annotations

from typing import Any, Iterator, List, Union

from marketlearn.algorithms.trees import tree_base as tb


class GeneralTree(tb._GeneralTreeBase):
    """Class representing general tree structure using linked representation"""

    class _Node:

        __slots__ = "_element", "_parent", "_children", "_total_children"

        def __init__(self, element: Any, parent=None, children=[]):
            self._element = element
            self._parent = parent
            self._children = children
            self._total_children = len(children) if children else 0

        def __len__(self):
            """returns total numbner of children in the node

            Returns
            -------
            int
                count of number of children in this node
            """
            return self._total_children

        def __repr__(self):
            return "Node({!r})".format(self._element)

    class Position(tb.Position):
        """Abstraction representing location of single element"""

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            """return element stored at position"""
            return self._node._element

        def __eq__(self, other: object):
            if not isinstance(other, Position):
                return NotImplemented
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
        node = self._validate(p)
        return len(node)

    def children(self, p: Position) -> Iterator[Position]:
        """generates iteration of p's children

        Parameters
        ----------
        p : Position
            [description]

        Yields
        -------
        Iterator[Position]
            [description]
        """
        node = self._validate(p)
        for child in node._children:
            yield self._make_position(child)

    def _add_root(self, data: Any) -> Position:
        """Adds root position to a tree

        Parameters
        ----------
        data : Any
            [description]

        Returns
        -------
        Position
            [description]

        Raises
        ------
        ValueError
            [description]
        """
        if self._root:
            raise ValueError("Root Exists")
        self._size = 1
        self._root = self._Node(data)
        return self._make_position(self._root)

    def _add_children(
        self, p: Position, children: List[Any]
    ) -> List[Position]:
        """Add children data into p's position

        Parameters
        ----------
        p : Position
            [description]
        children : List[Any]
            [description]

        Returns
        -------
        List[Position]
            list of positions representing p's children

        Raises
        ------
        ValueError
            if children data already exists
        """
        node = self._validate(p)
        if node._children is not None:
            raise ValueError("children already exist")
        node_children = [None] * len(children)
        for idx, child in enumerate(children):
            child_node = self._Node(child, parent=node)
            node_children[idx] = child_node
            self._size += 1

        # assign children to this node
        node._children = node_children
        return [self._make_position(n) for n in node._children]

    def positions(self, traversal: str = "preorder") -> Iterator[Position]:
        """Generates iterations of tree's positions

        Parameters
        ----------
        traversal : str, optional, default='inorder'
            one of preorder, postorder, breadthfirst

        Yields
        -------
        Iterator[Position]
            [description]
        """
        return getattr(self, traversal)()
