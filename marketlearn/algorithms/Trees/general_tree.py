"""Module Implements General Trees using linked lists

Author: Rajan Subramanian
Date: 11/11/2020
"""
from marketlearn.algorithms.Trees.tree_base import _GeneralTreeBase
from typing import Any


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
        """To represent contents of a node in trees"""

        __slots__ = '_element', '_parent'

        def __init__(self, element, parent=None, child=None):
            """default constructor used to initialize a node

            :param element: data contained in the node
            :type element: Any
            :param parent: the parent of the node element,
             defaults to None
            :type parent: node, optional
            :param child: child of the current node element,
             defaults to None
            :type child: node, optional
            """
            self._element = element
            self._parent = parent
            self._children = [] if child is None else [child]

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
        self._my_hash = {'preorder': self.preorder,
                         'postorder': self.postorder,
                         "breadthfirst": self.breadthfirst}

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
        return len(node._children) if node._children else 0

    def children(self, p):
        """returns iteration of p's children (or None if p is empty)

        :param p: positional object
        :type p: positional class
        """
        node = self._validate(p)
        if node._children is None:
            return None
        for child in node._children:
            yield self._make_position(child)

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

    def _addchild(self,
                  p: GTree.Position,
                  data: Any
                  ):
        """adds data as child of position p
           raise valueError if child already exists
           takes O(k) time where k is number of
           children of p

        :param p: [description]
        :type p: GTree.Position
        :param data: [description]
        :type data: Any
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

    def _replace(self, p, data):
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
            type: (str) tree traversal type, one of pre(post,in)order or breadthfirst
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
