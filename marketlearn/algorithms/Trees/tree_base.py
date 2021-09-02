from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Iterator, List, Optional

from marketlearn.algorithms.linked_collections import LinkedQueue


class Position(metaclass=ABCMeta):
    """Abstract Base Class representing position of an element"""

    @abstractmethod
    def element(self):
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

    def __repr__(self) -> str:
        return "Position({!r})".format(self.element())


class Tree(metaclass=ABCMeta):
    """Abstract Base Class representing tree structure"""

    def __iter__(self):
        """generates iteration of tree's elements"""
        for p in self.positions():
            yield p.element()

    @abstractmethod
    def root(self) -> Position:
        """return position representing T's root or None if Tree is empty

        Returns
        -------
        Position
            position of root
        """
        pass

    @abstractmethod
    def parent(self, p) -> Position:
        """return position representing p's parent (or None if p is root)

        Parameters
        ----------
        p : Position
            position of single element

        Returns
        -------
        Position
            position representing p's parent
        """
        pass

    @abstractmethod
    def num_children(self, p: Position) -> int:
        """return # of children that position p has

        Parameters
        ----------
        p : Position
            [description]

        Returns
        -------
        int
            [description]
        """
        pass

    @abstractmethod
    def children(self, p: Position) -> Iterator[Position]:
        """generates iteration of p's children

        Parameters
        ----------
        p : Position
            [description]
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """returns total number of elements in a tree

        Returns
        -------
        int
            number of elements in a tree
        """
        pass

    def is_root(self, p: Position) -> bool:
        """return true if position p represents root of tree

        Parameters
        ----------
        p : Position
            represents location of single element

        Returns
        -------
        bool
            True if positon p is root, false otherwise
        """
        return p == self.root()

    def is_leaf(self, p) -> bool:
        """returns True if position p does not have any children

        Parameters
        ----------
        p : Position
            represents location of single element

        Returns
        -------
        bool
            True if p does not have any children, false otherwise
        """
        return self.num_children(p) == 0

    @abstractmethod
    def positions(self) -> Iterator[Position]:
        """generates iteration of Trees positions

        Yields
        -------
        Iterator[Position]
            containing tree's positions
        """
        pass

    def is_empty(self) -> bool:
        """returns True if tree is empty

        Returns
        -------
        bool
            True if Tree is empty, False otherwise
        """
        return len(self) == 0

    def depth(self, p: Position) -> int:
        """Returns number of levels seperating position p from root

        Takes O(n) worse time

        Parameters
        ----------
        p : Position
            represents location of single element

        Returns
        -------
        int
            the depth of tree from root to position p
        """
        return 0 if self.is_root(p) else 1 + self.depth(self.parent(p))

    def _height1(self, p: Position) -> int:
        """Returns height of subtree rooted at position p

        Height of non empty tree T is is max of depth
        of its leaf positions
        Takes O(n^2) worse time

        Parameters
        ----------
        p : Position
            represents location of single element

        Returns
        -------
        int
            height of tree
        """
        return max(self.depth(p) for p in self.positions() if self.is_leaf(p))

    def _height2(self, p: Position) -> int:
        """returns height of subtree rooted at position p

        Parameters
        ----------
        p : Position
            represents position of single element

        Returns
        -------
        int
            height of tree
        """
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(p))

    def height(self, p: Optional[Position] = None):
        """returns height of subtree rooted at position p

        Parameters
        ----------
        p : Position, optional, default=None for full tree
            represents position of element

        Returns
        -------
        int
            height of tree, takes O(n) time
        """
        p = self.root() if p is None else p
        return self._height2(p)

    def breadthfirst(self) -> Iterator[Position]:
        """performs a breadth first tree traversal on tree
        Uses a LinkedQueue implementation"""
        if not self.is_empty():
            lq = LinkedQueue()
            lq.enqueue(self.root())
            while not lq.is_empty():
                p = lq.dequeue()
                yield p
                for c in self.children(p):
                    lq.enqueue(c)

    def preorder(self) -> Iterator[Position]:
        """generate a preorder iteration of positions in a tree

        Yields
        -------
        Iterator[Position]
            [preorder iteration of positions in a tree
        """
        if not self.is_empty():
            yield from self._subtree_preorder(self.root())

    def _subtree_preorder(self, p: Position) -> Iterator[Position]:
        """generate a preorder iteration of position in a subtree rooted at position p

        Parameters
        ----------
        p : Position
            iteration starts from this position

        Yields
        -------
        Iterator[Position]
            preorder iteration of positions in a subtree rooted at position p
        """
        # visit p first before visiting its subtrees
        yield p

        # for each child of p, do a preorder of its subtrees
        for c in self.children(p):
            yield from self._subtree_preorder(c)

    def _subtree_postorder(self, p: Position) -> Iterator[Position]:
        """generate a postorder iteration of position in a subtree rooted as position p

        Parameters
        ----------
        p : Position
            iteration starts from this position

        Yields
        -------
        Iterator[Position]
            [description]
        """
        for c in self.children(p):
            yield from self._subtree_postorder(c)
        yield p

    def postorder(self):
        """generate a postorder iteration of postions in a tree"""
        if not self.is_empty():
            yield from self._subtree_postorder(self.root())


class _BinaryTreeBase(Tree):
    """Abstract class representing binary tree methods"""

    @abstractmethod
    def left(self, p: Position) -> Optional[Position]:
        """return position representing p's left child

        return None if p does not have left child

        Parameters
        ----------
        p : Position
            represents the parent

        Returns
        -------
        Position
            represents p's left child or None if p has no child
        """
        pass

    @abstractmethod
    def right(self, p: Position) -> Optional[Position]:
        """return position representing p's right child
        return None if p does not have right child"

        Parameters
        ----------
        p : Position
            represents the parent

        Returns
        -------
        Union[Position, None]
            return p's right child or None if p has no child
        """
        pass

    def sibling(self, p: Position) -> Optional[Position]:
        """return position representing p's sibling (None if no siblings
        takes O(1) time

        Parameters
        ----------
        p : Position
            represents one of the siblings

        Returns
        -------
        Union[Position, None]
            position representing p's sibling or None if p has no siblings
        """
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            child = self.left(parent)
            if child:
                if p == child:
                    return self.right(parent)
                return self.left(parent)

    def children(self, p: Position) -> Iterator[Position]:
        """generates an iteration of p's children

        Parameters
        ----------
        p : Position
            the parent position

        Yields
        -------
        Iterator[Position]
            of p's children
        """
        left_child = self.left(p)
        if left_child:
            yield left_child
        right_child = self.right(p)
        if right_child:
            yield right_child

    def _subtree_inorder(self, p):

        if self.left(p) is not None:
            yield from self._subtree_inorder(self.left(p))
        yield p
        if self.right(p) is not None:
            yield from self._subtree_inorder(self.right(p))

    def inorder(self):
        if not self.is_empty():
            yield from self._subtree_inorder(self.root())


class _GeneralTreeBase(Tree):
    """Abstract class representing General tree methods"""

    def siblings(self, p: Position) -> Iterator[Position]:
        """return iterator representing p's sibling (None if no siblings
        takes O(1) time

        Parameters
        ----------
        p : Position
            represents one of the siblings

        Returns
        -------
        Union[Position, None]
            position representing p's sibling or None if p has no siblings
        """
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            for child in self.children(parent):
                if p != child:
                    yield child

    @abstractmethod
    def children(self, p: Position) -> Iterator[Position]:
        """generates an iteration of p's children

        Parameters
        ----------
        p : Position
            the parent position

        Yields
        -------
        Iterator[Position]
            of p's children
        """
        pass

    @abstractmethod
    def _add_children(self, p: Position, children: List[Any]):
        """adds children to p's position"""
        pass
