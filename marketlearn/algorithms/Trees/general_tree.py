"""Module Implements General Trees using linked lists

Author: Rajan Subramanian
Date: 11/11/2020
"""
from marketlearn.algorithms.Trees.tree_base import _GeneralTreeBase


class GTree(_GeneralTreeBase):
    """Class representing general tree structure using linked lists

    Parameters:
    None

    Attributes:
    root: (Node)        represents root of the binary tree
                        default set to None since its empty at time of creation
    size: (int)         length of tree
                        default to 0 since its empty at time of creation
    my_hash: (dict)     calls the tree traversal based on traversal type
    """
    # Nested Node Class
    class _Node:
        """To represent contents of a node in trees"""

        __slots__ = '_element', '_parent', '_children'

        def __init__(self, element, parent=None, children=None):
            self._element = element
            self._parent = parent
            self._children = children

    class Position(_GeneralTreeBase.Position):
        """Abstraction representing location of single element"""

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            """return element stored at position"""
            return self._node._element

        def __eq__(self, other):
            return type(other) is type(self) and other._node is self._node

    def _make_position(self, node):
        """Return Position's instance for a given node"""
        return self.Position(self, node) if node is not None else None

    def _validate(self, p):
        """return position's node or raise appropriate error if invalid"""
        if not isinstance(p, self.Position):
            raise("p must be proper Position type")
        if p._container is not self:
            raise ValueError("p does not belong to this container")
        if p._node._parent is p._node:
            raise ValueError("p is no longer valid")  # convention for deprecated nodes
        return p._node

    # general tree constructor
    def __init__(self):
        """Creates a initially empty general tree
           takes O(1) time
        """
        self._root = None
        self._size = 0

    def __len__(self):
        """returns total number of nodes in tree
           takes O(1) time
        """
        return self._size

    def root(self):
        """return root position of tree, return None if tree is empty
           takes O(1) time
        """
        return self._make_position(self._root)

    def parent(self, p):
        """return position representing p's parent (or None if p is root)

        :param p: positional object
        :type p: position of p's parent or None if p is empty
        """
        node = self._validate(p)
        return self._make_position(node._parent)
    
    def num_children(self, p):
        """return # of children that position p has
        takes O(1) time

        :param p: positional object
        :type p: positional class
        """
        node = self._validate(p)
        return len(node._children)

    def children(self, p):
        """returns iteration of p's children (or None if p is empty)

        :param p: positional object
        :type p: positional class
        """
        node = self._validate(p)
        if node._children is None:
            return None
        for child in node._children:
            yield child