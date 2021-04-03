"""Module Implements Binary Trees using linked lists

Author: Rajan Subramanian
Date: 11/11/2020
"""
from __future__ import annotations
from marketlearn.algorithms.trees import tree_base as tb
from typing import Union


class BinaryTree(tb._BinaryTreeBase):
    """Class representing binary tree structure using linked representation"""

    class _Node:

        __slots__ = "_element", "_parent", "_left", "_right"

        def __init__(self, element, parent=None, left=None, right=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right

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
        """return position's node or raise appropriate error if invalid"""
        if not isinstance(p, self.Position):
            raise ("p must be proper Position type")
        if p._container is not self:
            raise ValueError("p does not belong to this container")
        if p._node._parent is p._node:
            raise ValueError("p is no longer valid")  # convention for deprecated nodes
        return p._node

    # binary tree constructor
    def __init__(self):
        """Creates a initially empty binary tree"""
        self._root = None
        self._size = 0

    def __len__(self):
        """returns total number of nodes in a tree
        takes O(1) time"""
        return self._size

    def root(self) -> Position:
        """return root position of tree, return None if tree is empty
        takes O(1) time
        """
        return self._make_position(self._root)

    def parent(self, p) -> Union[Position, None]:
        """return position representing p's parent (or None if p is root)
        takes O(1) time
        """
        node = self._validate(p)
        return self._make_position(node._parent)

    def left(self, p) -> Union[Position, None]:
        """return position representing p's left child
        return None if p does not have left child
        takes O(1) time
        """
        node = self._validate(p)
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

    def _add_root(self, data):
        """place data at root of empty tree and return new position
        raise ValueError if tree nonempty
        takes O(1) time
        """
        if self._root is not None:
            raise ValueError("Root exists")
        self._size = 1
        self._root = self._Node(data)
        return self._make_position(self._root)

    def _add_left(self, p, data):
        """place data at left child of position p
        raise valueError if left child already exists
        takes O(1) time
        """
        node = self._validate(p)
        if node._left is not None:
            raise ValueError("left Child Exists")
        node._left = self._Node(data, parent=node)
        return self._make_position(node._left)

    def _add_right(self, p, data):
        """place data at right child of position p
        raise valueError if right child already exists
        takes O(1) time
        """
        node = self._validate(p)
        if node._right is not None:
            raise ValueError("right Child Exists")
        node._right = self._Node(data, parent=node)
        return self._make_position(node._right)

    def _replace(self, p, data):
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
        if child is not None:
            child._parent = node._parent  # child's grandparent becomes parent
        if node is self._root:
            self._root = child  # child becomes root
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

    def _attach(self, p, tree1=None, tree2=None):
        """Attach trees tree1 and tree2 as left and right subtrees fo external p
        takes O(1) time
        """
        node = self._validate(p)
        if not self.is_leaf(p):
            raise ValueError("position must be a leaf")

        # all trees must be of same type
        if not type(self) is type(tree1) is type(tree2):
            raise ValueError("Trees must match")
        self._size += len(tree1) + len(tree2)
        if not tree1.is_empty():
            tree1._root._parent = node
            node._left = tree1._root
            tree1._root = None  # set tree1 root instance to empty
            tree1._size = 0
        if not tree2.is_empty():
            tree2._root._parent = node
            node._right = tree2._root
            tree2._root = None  # set tree2 root instance to empty
            tree2.size = 0

    def preorder(self):
        """generate a preorder iteration of positions in a tree"""
        if not self.is_empty():
            for p in self._subtree_preorder(self.root()):
                yield p

    def _subtree_preorder(self, p):
        """generate a preorder iteration of positions in a subtree rooted at position p"""
        yield p  # visit p first before visiting its subtrees
        for c in self.children(p):
            for pos in self._subtree_preorder(c):
                yield pos

    def positions(self, traversal="inorder"):
        """generate iteration of trees positions
        params:
        type: (str) tree traversal type, one of pre(post,in)order or breadthfirst
                default set to inorder
        """

        return getattr(self, traversal)()

    def _subtree_postorder(self, p):
        """generate postorder iteration of positions in a subtree rooted at p"""
        for c in self.children(p):
            for pos in self._subtree_postorder(c):
                yield pos
        yield p

    def postorder(self):
        """generate a postorder iteration of postions in a tree"""
        if not self.is_empty():
            for p in self._subtree_postorder(self.root()):
                yield p

    def _subtree_inorder(self, p):

        if self.left(p) is not None:
            for pos in self._subtree_inorder(self.left(p)):
                yield pos
        yield p
        if self.right(p) is not None:
            for pos in self._subtree_inorder(self.right(p)):
                yield pos

    def inorder(self):
        if not self.is_empty():
            for p in self._subtree_inorder(self.root()):
                yield p
