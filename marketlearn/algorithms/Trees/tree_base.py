from marketlearn.algorithms.linked_collections import LinkedQueue


class Tree:
    """Abstract Base Class representing tree structure"""

    class Position:
        """Abstraction representing location of a single element"""

        def element(self):
            """Returns elements stored in this position

            Raises
            ------
            NotImplementedError
                must be implemented by subclass
            """
            raise NotImplementedError("must be implemented by subclass")

        def __eq__(self, other):
            """returns True if other position represents same location

            Parameters
            ----------
            other : Position
                location of element

            Raises
            ------
            NotImplementedError
                must be implemented by subclass
            """
            raise NotImplementedError("must be implemented by subclass")

        def __ne__(self, other):
            """returns True if other does not represent the same location

            Parameters
            ----------
            other : Position
                position of single element

            Returns
            -------
            bool
                return True if other does not represent same location
                False otherwsie
            """
            return not (self == other)

    # abstract methods that concrete subclass must support--------------------
    def root(self):
        """return position representing tree's root or None if empty

        Raises
        ------
        NotImplementedError
            must be implemented in subclass
        """
        raise NotImplementedError("must be implemented by subclass")

    def parent(self, p):
        """return position representing p's parent (or None if p is root)

        Parameters
        ----------
        p : Position
            represent location of element

        Raises
        ------
        NotImplementedError
            must be implemented in subclass
        """
        raise NotImplementedError("must be inmplemented in subclass")

    def num_children(self, p):
        """return # of children that position p has

        Parameters
        ----------
        p : Position
            represents location of single element

        Raises
        ------
        NotImplementedError
            must be implemented in subclass
        """
        raise NotImplementedError("must be implemented by subclass")

    def children(self, p):
        """generate iteration of p's children

        Parameters
        ----------
        p : Position
            represents location of single element

        Raises
        ------
        NotImplementedError
            must be implemented in subclass
        """
        raise NotImplementedError("must be implemented by subclass")

    def __len__(self):
        """return total number of elements in a tree

        Raises
        ------
        NotImplementedError
            must be implemented in subclass
        """
        raise NotImplementedError("must be implemented by subclass")

    def is_root(self, p):
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

    def is_leaf(self, p):
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

    def positions(self):
        """generates iterations of tree's positions

        Raises
        ------
        NotImplementedError
            must be implemented in subclass
        """
        raise NotImplementedError("must be implemented by subclass")

    def is_empty(self):
        """returns True if tree is empty

        Returns
        -------
        bool
            True if Tree is empty, False otherwise
        """
        return len(self) == 0

    def depth(self, p):
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

    def _height1(self, p):
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

    def _height2(self, p):
        """returns height of subtree rooted at position p
        takes O(n) time
        """
        return (
            0
            if self.is_leaf(p)
            else 1 + max(self._height2(c) for c in self.children(p))
        )

    def height(self, p=None):
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

    def __iter__(self):
        """generates iteration of tree's elements"""
        for p in self.positions():
            yield p.element()

    def breadthfirst(self):
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


class _BinaryTreeBase(Tree):
    """Abstract class representing binary tree methods"""

    def left(self, p):
        """return position representing p's left child

        return None if p does not have left child

        :param p: position of p (parent)
        :type p: PositionalObj
        :raises NotImplementedError: needs to be implemented in subclass
        """
        raise NotImplementedError("must be implemented by subclass")

    def right(self, p):
        """return position representing p's right child
        return None if p does not have right child"""

        raise NotImplementedError("must be implemented by subclass")

    def sibling(self, p):
        """return position representing p's sibling (None if no siblings
        takes O(1) time
        """
        parent = self.parent(p)
        if parent is None:
            return None
        else:
            if p == self.left(parent):
                return self.right(parent)
            return self.left(parent)

    def children(self, p):
        """generates an iteration of p's children"""
        if self.left(p) is not None:
            yield self.left(p)
        if self.right(p) is not None:
            yield self.right(p)


class _GeneralTreeBase(Tree):
    """Abstract Base Class representing tree structure"""

    def children(self, p):
        """generates an iteration of p's children

        :param p: position of parent
        :type p: positional object
        """
        raise NotImplementedError("Must be Implemented by subclass")