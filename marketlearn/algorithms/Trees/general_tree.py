"""Module Implements General Trees using linked lists

Author: Rajan Subramanian
Date: 11/11/2020
"""
from marketlearn.algorithms.trees.tree_base import _GeneralTreeBase
from typing import Any, Iterator


class GTree(_GeneralTreeBase):
    """Class representing general tree structure using linked lists

    Parameters:
    None

    Attributes:
    root: (Node)        represents root of the general tree
                        default set to None since its empty at time of creation
    size: (int)         length of tree
                        default to 0 since its empty at time of creation
    my_hash: (dict)     calls the tree traversal based on traversal type
    """
    # Nested Node Class
    class _Node:
        """To represent contents of a node in trees

        Parameters
            ----------
            element : Any
                data to be inserted into node
            parent : {Node, optional}, default=None
                parent of current node, by default None
            child : {Node, optional}, deault=None
                child of current node, by default None
        """
        # - light weight slots for trees
        __slots__ = '_element', '_parent'

        def __init__(self, element: Any, parent=None, child=None):
            """default constructor used to initialize a node"""
            self._element = element
            self._parent = parent
            self._children = [] if child is None else [child]

    class Position(_GeneralTreeBase.Position):
        """Abstraction representing location of single element"""

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            """return element stored at position

            Returns
            -------
            Any
                element stored at a position
            """
            return self._node._element

        def __eq__(self, other):
            """tests if two positions are equal

            Parameters
            ----------
            other : Position
                the position we are comparing to

            Returns
            -------
            bool
                True if two positions are the same and they
                contain the same node
            """
            return type(other) is type(self) and other._node is self._node

    def _make_position(self, node):
        """Return Position's instance for a given node

        Parameters
        ----------
        node : Node
            [description]

        Returns
        -------
        Position
            position's instance for a given node
        """
        return self.Position(self, node) if node is not None else None

    def _validate(self, p):
        """return position's node or raise appropriate error if invalid

        Parameters
        ----------
        p : [type]
            position whose node will be returned

        Returns
        -------
        Node
            the node of the position

        Raises
        ------
        ValueError
            if p is not a Position object
        ValueError
            if p does not belong to same container
        ValueError
            if p's node is same as p's parent node
        """
        if not isinstance(p, self.Position):
            raise("p must be proper Position type")
        if p._container is not self:
            raise ValueError("p does not belong to this container")
        if p._node._parent is p._node:
            # convention for deprecated nodes
            raise ValueError("p is no longer valid")
        return p._node

    # general tree constructor
    def __init__(self):
        """Creates a initially empty general tree
           takes O(1) time
        """
        self._root = None
        self._size = 0
        self._my_hash = {'preorder': self.preorder,
                         'postorder': self.postorder,
                         "breadthfirst": self.breadthfirst}

    def __len__(self):
        """returns total number of nodes in a tree

        Returns
        -------
        int
            total number of nodes in a tree
            takes O(1) time
        """
        return self._size

    def root(self):
        """return root of tree's position, or None if tree is empty

        Returns
        -------
        Position
            the root position or None if tree is empty
        """
        return self._make_position(self._root)

    def parent(self, p):
        """return position representing p's parent (or None if p is root)

        Parameters
        ----------
        p : Position
            position of interest

        Returns
        -------
        Position
            position of p's parent
        """
        node = self._validate(p)
        return self._make_position(node._parent)

    def num_children(self, p):
        """return total children that position p has

        Parameters
        ----------
        p : Position
            position of interest

        Returns
        -------
        int
            the total number of p's children
        """
        node = self._validate(p)
        return len(node._children) if node._children else 0

    def children(self, p):
        """returns iteration of p's children (or None if p is empty)

        Parameters
        ----------
        p : Position
            position of interest

        Returns
        -------
        None
            if p has no children

        Yields
        -------
        Iterator
            an iterator of p's children
        """
        """
        node = self._validate(p)
        if node._children is None:
            return None
        for child in node._children:
            yield self._make_position(child)"""
        pass

    def _addroot(self, data: Any):
        """adds data to root of empty tree and return new position

        :param data: [description]
        :type data: Any
        """
        if self._root is not None:
            raise ValueError("Root already exists")
        self._size = 1
        self._root = self._Node(data)
        return self._make_position(self._root)

    def _addchild(self, p, data: Any):
        """adds data as child to position p

        Parameters
        ----------
        p : Position
            current position to which child is added
        data : Any
            data to be used into child node

        Returns
        -------
        Position
            child position

        Raises
        ------
        ValueError
            if position p already has this child
        """
        node = self._validate(p)
        child_node = self._Node(data)

        # check if child is p's child
        if child_node._element in node._children:
            raise ValueError("position already has this child")

        # associate child's parent to p's node
        self._size += 1
        child_node._parent = node

        # add child to p's family and return child's position
        node._children.append(child_node)
        return self._make_position(child_node)

    def _replace(self, p, data: Any):
        """replace data at position p with data and returns old data
            takes O(1) time
        """
        node = self._validate(p)
        old = node._element
        node._element = data
        return old

    def positions(self, type='postorder'):
        """generate iteration of trees positions
            params:
            type: (str) tree traversal type, one of pre(post,in)order
            or breadthfirst
                    default set to inorder
        """
        if type not in ('preorder', 'postorder', 'breadthfirst'):
            raise AttributeError()
        return self._my_hash[type]()

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

    def preorder(self):
        """generate a preorder iteration of positions in a tree"""
        if not self.is_empty():
            for p in self._subtree_preorder(self.root()):
                yield p

    def _subtree_preorder(self, p):
        """generate a preorder iteration of positions in a subtree rooted at position p
            """
        yield p  # visit p first before visiting its subtrees
        for c in self.children(p):
            for pos in self._subtree_preorder(c):
                yield pos
