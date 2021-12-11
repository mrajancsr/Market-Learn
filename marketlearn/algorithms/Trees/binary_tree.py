"""Module Implements Binary Trees using linked lists

Author: Rajan Subramanian
Date: 11/11/2020
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Union

from marketlearn.algorithms.trees import tree_base as tb


@dataclass
class Node:
    element: Any
    parent: Optional[Node] = None
    left: Optional[Node] = None
    right: Optional[Node] = None


@dataclass
class Position(tb.Position):
    """Abstration representing location of single element"""

    container: BinaryTree
    node: Node

    def element(self) -> Any:
        """return element stored at position"""
        return self.node.element

    def __eq__(self, other: Position) -> bool:
        return type(other) is type(self) and other.node is self.node


@dataclass
class BinaryTree(tb._BinaryTreeBase):
    """Class representing binary tree structure using linked representation"""

    _root: Optional[Node] = field(init=False)
    _size: int = 0

    def __post_init__(self) -> None:
        self._root = None

    def __len__(self):
        """returns total number of nodes in a tree"""
        return self._size

    def _make_position(self, node: Optional[Node]) -> Optional[Position]:
        """Return Position's instance for a given node"""
        return Position(self, node) if node is not None else None

    def _validate(self, p: Position) -> Optional[Node]:
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
        if not isinstance(p, Position):
            raise TypeError("p must be proper Position type")
        if p.container is not self:
            raise TypeError("p does not belong to this container")

        # convention for deprecated nodes
        if p.node.parent is p.node:
            raise ValueError("p is no longer valid")
        return p.node

    def root(self) -> Optional[Position]:
        """return root position of tree, return None if tree is empty"""
        return self._make_position(self._root)

    def parent(self, p: Position) -> Optional[Position]:
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
        if node is not None:
            return self._make_position(node.parent)

    def left(self, p) -> Union[Position, None]:
        """return position representing p's left child
        return None if p does not have left child
        takes O(1) time
        """
        node = self._validate(p)
        if node is not None:
            return self._make_position(node._left)

    def right(self, p: Position) -> Union[Position, None]:
        """return position representing p's right child

        takes O(1) time

        Parameters
        ----------
        p : Position
            represents the parent

        Returns
        -------
        Union[Position, None]
            position representing p's right child or
            None if p doens't have any children
        """

        node = self._validate(p)
        return self._make_position(node._right)

    def num_children(self, p: Position) -> int:
        """return count of total children of position p

        Parameters
        ----------
        p : Position
            represents the parent position

        Returns
        -------
        int
            count of total children at position p
        """
        node = self._validate(p)
        count = 0
        if node._left is not None:
            count += 1
        if node._right is not None:
            count += 1
        return count

    def _add_root(self, data: Any):
        """place data at root of empty tree and return new position
        raise ValueError if tree nonempty
        takes O(1) time
        """
        if self._root is not None:
            raise ValueError("Root exists")
        self._size = 1
        self._root = self._Node(data)
        return self._make_position(self._root)

    def _add_left(self, p: Position, data: Any):
        """place data at left child of position p
        raise valueError if left child already exists
        takes O(1) time
        """
        node = self._validate(p)
        if node._left is not None:
            raise ValueError("left Child Exists")
        node._left = self._Node(data, parent=node)
        return self._make_position(node._left)

    def _add_right(self, p: Position, data: Any):
        """place data at right child of position p
        raise valueError if right child already exists
        takes O(1) time
        """
        node = self._validate(p)
        if node._right is not None:
            raise ValueError("right Child Exists")
        node._right = self._Node(data, parent=node)
        return self._make_position(node._right)

    def _replace(self, p: Position, data: Any):
        """replace data at position p with data and returns old data
        takes O(1) time
        """
        node = self._validate(p)
        old = node._element
        node._element = data
        return old

    def _delete(self, p: Position):
        """delete node at position p and replace it with its child, if any

        Takes O(1) time

        Parameters
        ----------
        p : Position
            position of node to be removed

        Returns
        -------
        Any
            element in the node

        Raises
        ------
        ValueError
            if p's total children is 2 or p is invalid position
        """
        node = self._validate(p)
        if self.num_children(p) == 2:
            raise ValueError("p has two children")

        # get one of the children
        child = node._left if node._left else node._right

        # replace child's parent with his grandpa if not empty
        if child is not None:
            child._parent = node._parent

        # edge case
        if node is self._root:
            self._root = child
        else:
            parent = node._parent
            if node is parent._left:
                parent._left = child
            else:
                parent._right = child
        self._size -= 1

        # deprecate the node (convention)
        node._parent = node
        return node._element

    def _attach(self, p: Position, tree1: BinaryTree, tree2: BinaryTree) -> None:
        """Attach trees tree1 and tree2 as left and right subtrees fo external p
        takes O(1) time

        Parameters
        ----------
        p : Position
            leaf position where tree1 and tree2 are attached to
        tree1 : BinaryTree
            left subtree of position p
        tree2 : BinaryTree
            right subtree of position p

        Raises
        ------
        ValueError
            position is not a leaf
        TypeError
            tree1 and tree2 is not BinaryTree
        """
        node = self._validate(p)
        if not self.is_leaf(p):
            raise ValueError("position must be a leaf")

        # all trees must be of same type
        if not type(self) is type(tree1) is type(tree2):
            raise TypeError("Trees must match")

        # size of new tree is sum of positions of two trees
        self._size += len(tree1) + len(tree2)

        # attach the first subtree
        if not tree1.is_empty():
            tree1._root._parent = node
            node._left = tree1._root

            # convention for deprecated nodes
            tree1._root = None
            tree1._size = 0

        # attach the right subtree
        if not tree2.is_empty():
            tree2._root._parent = node
            node._right = tree2._root

            # convention for deprecated nodes
            tree2._root = None
            tree2.size = 0

    def positions(self, traversal="inorder") -> Iterator[Position]:
        """generates iterations of tree's positions

        Parameters
        ----------
        traversal : str, optional, default='inorder'
            one of inorder, preorder, postorder, breadthfirst

        Yields
        -------
        Iterator[Position]
            iteration of tree's positions
        """

        return getattr(self, traversal)()
