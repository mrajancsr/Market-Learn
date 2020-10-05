"""linear collections of Stack, Queue and Deque implemented via linked List
Takes O(1) time for all insertion, removal operations
-O(n) time complexity for traversing the ADTs to print the elements
"""

class LinkedStack:
    """LIFO Stack Implementation using a singly linked list
        Takes O(1) time

    params:
    None

    Attributes:
    start_node:  (Node) represents head of the linked list
                        default set to None since its empty at time of creation
    """
    class Node:
        """
        Nested Node Class.
        Create a node to store value and reference for LinkedList
        Takes O(1) time

        params:
        element: represents element of the node
        nref:    next reference of the node
        """
        __slots__ = 'element', 'nref'  # streamline memory usage

        def __init__(self, data, reference=None):
            self.element = data
            self.nref = reference

    def __init__(self):
        self.start_node = None
        self.size = 0

    def __len__(self):
        """Returns number of elements in a stack
            takes O(1) time"""
        return self.size

    def is_empty(self):
        """Return True if Stack is empty
            takes O(1) time"""
        return self.size == 0

    def pop(self):
        """Removes and returns element from top of the stack (LIFO)
            takes O(1) time
        """
        if self.is_empty():
            raise("stack has no elements to delete")
        element = self.start_node.element
        # delete by assigning next reference of start node to start node
        self.start_node = self.start_node.nref
        self.size -= 1
        return element

    def push(self, data):
        """Adds element to top of the Stack
            Takes O(1) time
        """
        new_node = self.Node(data)  # create a node
        new_node.nref = self.start_node  # new nodes next ref is old nodes start
        self.start_node = new_node  # new node is now the start_node
        self.size += 1

    def peek(self):
        """Returns but does not remove element from top of Stack
            takes O(1) time
        """
        if self.is_empty():
            raise("stack is empty")
        return self.start_node.element

    def traverse(self):
        """Traverses a Singly Linked List
            Takes O(n) time
        """
        if self.start_node is None:
            print("list has no elements")
        else:
            n = self.start_node
            while n is not None:
                print(n.element, " ")
                n = n.nref


class LinkedQueue:
    """FIFO Queue Implementation using a singly linked list
        Takes O(1) time for all operations
    params:
    None

    Attributes:
    start_node:  (Node) represents head of the linked list
                        default set to None since its empty at time of creation
    end_node:    (Node) represents tail of the linked list
                        default set to None since its empty at time of creation

    """
    class Node:
        """
        Nested Node Class.
        Create a node to store value and reference for LinkedList
        Takes O(1) time

        params:
        element: represents element of the node
        nref:    next reference of the node

        """
        __slots__ = 'element', 'nref'  # streamline memory usage

        def __init__(self, data, reference=None):
            self.element = data
            self.nref = reference

    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.size = 0

    def __len__(self):
        """Returns the number of items in a Queue
            takes O(1) time
        """
        return self.size 

    def is_empty(self):
        """Returns True if Queue is empty
            takes O(1) time
        """
        return self.size == 0

    def enqueue(self, data):
        """Adds element to back of the Queue
            Takes O(1) time
        """
        new_node = self.Node(data)  # create a node
        if self.is_empty():
            self.start_node = new_node
            self.end_node = new_node # creating a circular reference
            self.size += 1
            return
        # else set the next reference of end node
        self.end_node.nref = new_node
        self.end_node = new_node # update ref of end node to new node
        self.size += 1

    def dequeue(self):
        """removes item at top of the Queue
            takes O(1) time
        """
        if self.is_empty():
            raise("Queue has no elements to delete")
        element = self.start_node.element
        # delete by assigning next reference of start node to start node
        self.start_node = self.start_node.nref
        self.size -= 1
        if self.is_empty():  # special case for one element
            self.end_node = None  # enforce tail node to None
        return element

    def traverse(self):
        """Traverses a Singly Linked List
            Takes O(n) time
        """
        if self.start_node is None:
            print("list has no elements")
        else:
            n = self.start_node
            while n is not None:
                print(n.element, " ")
                n = n.nref



class LinkedDeque:
    """Deque Implementation using a Doubly Linked List
        Takes O(1) time for all operations
    params:
    None

    Attributes:
    start_node:  (Node) represents head of the dlinked list
                        default set to None since its empty at time of creation
    end_node:    (Node) represents tail of the linked list
                        default set to None since its empty at time of creation

    """
    class Node:
        """
        Nested Node Class.
        Create a node to store value and 2 references for LinkedList
        Takes O(1) time

        params:
        element: represents element of the node
        nref:    next reference of the node
        pref:    previous reference of the node

        """

        def __init__(self, data, next_ref=None, prev_ref=None):
            self.element = data
            self.nref = next_ref
            self.pref = prev_ref

    def __init__(self):
        self.start_node = None
        self.end_node = None
        self.size = 0

    def __len__(self):
        return self.size

    def is_empty(self):
        """returns True if Deque is empty"""
        return self.size == 0

    def add_front(self,data):
        """inserets data at start of Deque
        takes O(1) time
        """
        if self.is_empty():
            new_node = self.Node(data)
            self.start_node = new_node 
            self.end_node = new_node # creating a circular reference
            self.size += 1
            return 
        new_node = self.Node(data) 
        new_node.nref = self.start_node  # make next reference of new node to current node
        self.start_node.pref = new_node # set previous reference of currenet node to new node
        self.start_node = new_node
        self.size += 1

    def remove_front(self):
        """deletes a node from the start of Deque 
            and returns the element
            takes O(1) time
        """
        if self.is_empty():
            raise("Deque is empty")
        element = self.start_node.element
        if self.start_node.nref is None:
            self.start_node = None  # if only one element, delete this
            self.size -= 1
            return element
        self.start_node = self.start_node.nref
        self.start_node.pref = None
        self.size -= 1
        # one element case after deleting previous node
        if self.is_empty():
            self.end_node = None
        return element

    def add_rear(self, data):
        """Adds element to back of the Deck
            Takes O(1) time
        """
        new_node = self.Node(data)  # create a node
        if self.is_empty():
            self.start_node = new_node
            self.end_node = new_node # creating a circular reference
            self.size += 1
            return
        # else set the next reference of end node
        self.end_node.nref = new_node
        new_node.pref = self.end_node
        self.end_node = new_node # update ref of end node to new node
        self.size += 1

    def remove_rear(self):
        """removes element at back of the Deck
            Takes O(1) time
        """
        # to do.  not correctly removing rear
        if self.is_empty():
            raise("Deck is empty")

        element = self.end_node.element

        self.end_node = self.end_node.pref
        self.end_node.nref = None
        if self.is_empty():
            self.end_node = None
        self.size -= 1
        return element

    def traverse(self):
        """Traverses a Singly Linked List
            Takes O(n) time
        """
        if self.start_node is None:
            print("list has no elements")
        else:
            n = self.start_node
            while n is not None:
                print(n.element, " ")
                n = n.nref

from linked_lists.linkedlist import DoublyLinkedBase
class PositionalList(DoublyLinkedBase):
    """Sequential Container of elements allowing positional access
        Built out of DoublyLinkedList
    """

    # -----------------nested Position class -----------------------
    class Position:
        """An abstraction representating position of a single element"""
        def __init__(self, container, node):
            self._container = container 
            self._node = node 

        def element(self):
            return self._node._element

        def __eq__(self, other):
            """Return True if other is a Position representing same location"""
            return type(other) is type(self) and other._node is self._node 

        def __ne__(self):
            """return true if other does not represent the same location"""
            return not (self == other)

    def _validate(self, p):
        """return position's node or raise appropriate error if invalid"""
        if not isinstance(p, self.Position): raise("p must be proper Position type")
        if p._container is not self: raise ValueError("p does not belong to this container")
        if p._node._nref is None: raise ValueError("p is no longer valid")
        return p._node

    def _make_position(self, node):
        """Return Position's instance for a given node (or None if sentinel)"""
        if node in (self._start_node, self._end_node): return None 
        else: return self.Position(self, node)
# -----------Accessors ------------#
    def first(self):
        """return the first position in the list or None if empty"""
        return self._make_position(self._start_node._nref)

    def last(self):
        """return last position in the list or None if empty"""
        return self._make_position(self._end_node._pref)

    def before(self, p):
        """return position just before positon p or None if p is first"""
        node = self._validate(p)
        return self._make_position(node._pref)

    def after(self, p):
        """return position just after position p or None if p is last"""
        node = self._validate(p)
        return self._make_position(node._nref)

    def __iter__(self):
        """generate an iterator of elements in the list"""
        cursor = self.first()
        while cursor is not None:
            yield cursor.element()
            cursor = self.after(cursor)
    # function overriden
    def _insert_between(self, data, node1, node2):
        """add element between existing nodes and return new position"""
        node = super()._insert_between(data, node1, node2)
        return self._make_position(node)

    def add_first(self, data):
        return self._insert_between(data, self._start_node, self._start_node._nref)

    def add_last(self, data):
        return self._insert_between(data, self._end_node._pref, self._end_node)

    def add_before(self, p, data):
        pnode = self._validate(p)
        return self._insert_between(data, pnode.pref, pnode)

    def add_after(self, p, data):
        pnode = self._validate(p)
        return self._insert_between(data, pnode, pnode._nref)

    def delete(self, p):
        """remove and return elemenet and position p"""
        node = self._validate(p)
        return self._delete_node(node)

    def replace(self, p, data):
        """replace element at position p with data"""
        node = self._validate(p)
        old_value = node._element
        node._element = data 
        return old_value






