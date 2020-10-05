

class Tree:
    """Abstract Base Class representing tree structure"""
    class Position:
        """Abstraction representing location of a single element"""

        def element(self):
            """Returns elements stored in this position"""
            raise NotImplemented("must be implemented by subclass")

        def __eq__(self, other):
            """returns True if other position represents same location"""
            raise NotImplemented("must be implemented by subclass")

        def __ne__(self, other):
            """returns Trueif other does not represent the same location"""
            return not (self == other)

    # abstract methods that concrete subclass must support--------------------
    def root(self):
        """return position representing tree's root or None if empty"""
        raise NotImplemented("must be implemented by subclass")

    def parent(self, p):
        """return position representing p's parent (or None if p is root)"""
        raise NotImplemented("must be implemented by subclass")

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
        """ returns height of subtree rooted at position p
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
        import os
        cwd = os.getcwd()
        os.chdir('/Users/raj/Documents/algorithms_in_python/linked_lists/')
        from linked_collections import LinkedQueue
        os.chdir(cwd)  # change to cwd
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


class BinaryTree(_BinaryTreeBase):
    """Class representing binary tree structure using linked representation
    
    Attributes:
    root:  (Node)       represents root of the binary tree
                        default set to None since its empty at time of creation
    size:   (int)       length of the tree
                        default to 0 since its empty at time of creation

    my_hash (dict)      calls the tree traversal based on traversal type
                        traversal type one of pre(in,post) order and breadthfirst
    """

    class _Node:

        __slots__ = '_element', '_parent', '_left', '_right'

        def __init__(self, element, parent=None, left=None, right=None):
            self._element = element
            self._parent = parent
            self._left = left
            self._right = right

    class Position(_BinaryTreeBase.Position):
        """Abstraction representing location of single element"""

        def __init__(self, container, node):
            self._container = container
            self._node = node

        def element(self):
            """return element stored at position """
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


    # binary tree constructor
    def __init__(self):
        """Creates a initially empty binary tree
            takes O(1) time
        """
        self._root = None
        self._size = 0
        self._my_hash = {'preorder':self.preorder, 'postorder': self.postorder, 'inorder': self.inorder,
        "breadthfirst": self.breadthfirst}

    def __len__(self):
        """returns total number of nodes in a tree
            takes O(1) time """
        return self._size

    def root(self):
        """return root position of tree, return None if tree is empty
            takes O(1) time
        """
        return self._make_position(self._root)

    def parent(self, p):
        """return position representing p's parent (or None if p is root)
            takes O(1) time 
        """
        node = self._validate(p)
        return self._make_position(node._parent)

    def left(self, p):
        """return position representing p's left child
            return None if p does not have left child
            takes O(1) time
        """
        node = self._validate(p)
        return self._make_position(node._left)

    def right(self, p):
        """return position representing p's right child
            return None if p does not have right child
            takes O(1) time
        """

        node = self._validate(p)
        return self._make_position(node._right)

    def num_children(self, p):
        """return # of children that position p has
            takes O(1) time
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

    def _delete(self, p):
        """delete node at position p and replace it with its child, if any
            return the data stored at position p
            raise ValueError if p is invalid or p has 2 children
            takes O(1) time
        """
        node = self._validate(p)
        if self.num_children(p) == 2: 
            raise ValueError("p has two children")
        child = node._left if node._left else node._right  # might be None 
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
        node._parent = node  # convention for deprecated nodes
        return node._element

    def _attach(self, p, tree1=None, tree2=None):
        """Attach trees tree1 and tree2 as left and right subtrees fo external p
            takes O(1) time
        """
        node = self._validate(p)
        if not self.is_leaf(p):
            raise ValueError("position must be a leaf")
        if not type(self) is type(tree1) is type(tree2):  # all trees must be of same type
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
        """generate a preorder iteration of positions in a subtree rooted at position p
            """
        yield p  # visit p first before visiting its subtrees
        for c in self.children(p):
            for pos in self._subtree_preorder(c):
                yield pos

    def positions(self, type='inorder'):
        """generate iteration of trees positions
            params:
            type: (str) tree traversal type, one of pre(post,in)order or breadthfirst
                    default set to inorder
        """
        if type not in set(('preorder','postorder','inorder','breadthfirst')):
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
