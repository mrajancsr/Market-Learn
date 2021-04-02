from marketlearn.algorithms.linked_collections import LinkedQueue


class Tree:
    """Abstract Base Class representing tree structure"""

    class Position:
        """Abstraction representing location of a single element"""

        def element(self):
            """Returns elements stored in this position

            Raises
            ------
            NotImplemented
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
            NotImplemented
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
        """return # of children that position p has"""
        raise NotImplemented("must be implemented by subclass")

    def children(self, p):
        """generate iteration of p's children"""
        raise NotImplemented("must be implemented by subclass")

    def __len__(self):
        """return total number of elements in a tree"""
        raise NotImplemented("must be implemented by subclass")

    def is_root(self, p):
        """return true if position p represents root of tree
        takes O(1) time
        """
        return p == self.root()

    def is_leaf(self, p):
        """return true if position p does not have any children
        takes O(1) time since num_children takes O(1) time
        """
        return self.num_children(p) == 0

    def positions(self, type=None):
        """generates iterations of tree's positions"""
        raise NotImplemented("must be implemented by subclass")

    def is_empty(self):
        """Return True if tree is empty
        takes O(1) time
        """
        return len(self) == 0

    def depth(self, p):
        """returns # of levels seperating position p from root
        takes O(n) worse time
        """
        if self.is_root(p):
            return 0
        else:
            return 1 + self.depth(self.parent(p))

    def _height1(self, p):
        """returns height of subtree rooted at position p
        height of non empty tree T is is max of depth
        of its leaf positions
        Takes O(n^2) worse time
        """
        return max(self.depth(p) for p in self.positions() if self.is_leaf(p))

    def _height2(self, p):
        """returns height of subtre rooted at position p
        takes O(n) time
        """
        if self.is_leaf(p):
            return 0
        else:
            return 1 + max(self._height2(c) for c in self.children(p))

    def height(self, p=None):
        """returns height of subtree rooted at position p
        params: p (position p, default = None for entire tree height)
        takes O(n) time
        """
        if p is None:
            p = self.root()
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

        raise NotImplemented("must be implemented by subclass")

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
        raise NotImplemented("Must be Implemented by subclass")