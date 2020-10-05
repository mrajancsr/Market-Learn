"""class for singly and doubly linked lists"""

class SinglyLinkedList:
    """Create a new Singly Linked List
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

        def __init__(self, data, reference=None):
            self.element = data
            self.nref = reference

    def __init__(self):
        self.start_node = None
        self.size = 0

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

    def insert_at_start(self, data):
        """inserts data at start of the list
        takes O(1) time
        """
        new_node = self.Node(data)  # create a node
        new_node.nref = self.start_node # new nodes next reference is old nodes start
        self.start_node = new_node  # new node is now the start_node
        self.size += 1
    
    def insert_at_end(self,data):
        """inserts value at end of the list
        if list is empty, takes O(1) time.  Otherwise:
        O(n) time
        """
        new_node = self.Node(data)
        if self.start_node is None:  # if list is empty
            self.start_node = new_node  # then start node is the new node
            self.size += 1
            return
        n = self.start_node
        # iterate until last element
        while n.nref is not None:
            n = n.nref
        n.nref = new_node  # set the next reference to point to new node
        self.size += 1
    
    def insert_after_item(self,x,data):
        """inserts value after item x is found
        if found, takes O(n-k), Otherwise, O(n)
        """
        n = self.start_node
        while n:  # iterate until x is found
            if n.element == x:
                break
            n = n.nref
        if n is None:
            print("item not found in list")
        else: # when element is found
            new_node = self.Node(data)   # create a new node
            new_node.nref = n.nref  # set the new nodes next ref to current nodes next ref
            n.nref = new_node # set the current nodes next ref to new node
            self.size += 1
    
    def insert_before_value(self, value,cdata):
        """Inserts data before value
        params:
        value:      item you are looking for
        data:   data to insert before item
        """
        if self.start_node is None:
            print("list has no elements")
            return 
        # if element is found in first node
        if self.start_node.element == value:
            new_node = self.Node(data)
            new_node.nref = self.start_node
            self.start_node = new_node
            self.size += 1
            return
        n = self.start_node
        # iterate until next node contains element
        while n.nref is not None:
            if n.nref.element == value:
                break
            n = n.nref
        if n.nref is None:
            print("value not found is list")
        else:
            new_node = self.Node(data)
            new_node.nref = n.nref  # new nodes next reference is previous nodes next
            n.nref = new_node       # previous node's next reference is now the new node
            self.size += 1

    def insert_at_index(self, index, data):
        if index == 1:
            new_node = self.Node(data)
            new_node.nref = self.start_node
            self.start_node = new_node
            self.size += 1
            return
        n = self.start_node
        # similar to insert after element
        while n and index > 1:
            n = n.nref
            index -= 1
        if n is None:
            print("Index out of bounds")
        else:
            new_node = self.Node(data)
            new_node.nref = n.nref
            n.nref = new_node
            self.size += 1
    def count_positions(self):
        """Returns the count of nodes in Linked List"""
        return self.size
    def search(self, value):
        """searches a linked list for given value

        params:
        value:  item to look for

        Returns:
        bool
        """
        if self.start_node is None:
            print("list has non elements")
            return
        n = self.start_node
        while n:
            if n.element == x: return True
            n = n.nref
        return False

    def reverse(self):
        n = self.start_node # current node
        prev_node = None  
        next_node = None 
        while n: # assume first node case...
            next_node = n.nref  # save the next reference of current node
            n.nref = prev_node  # replace next reference to prev node
            prev_node = n  # prev_node now is the current node
            n = next_node  # increment over to next node
        self.start_node = prev_node

    def delete_from_start(self):
        if self.start_node is None:
            print("list has no elements to delete")
            return
        # delete by assining next reference of start node to start node
        self.start_node = self.start_node.nref
    
    def delete_from_end(self):
        if self.start_node is None:
            print("list has no elements to delete")
            return 
        n = self.start_node
        while n.nref.nref:
            n = n.nref
        n.nref = None

    def delete_by_value(self,value):
        """deletes node that contains the value
        params:
        value:  item to look for in the node to delete
        """
        if self.start_node is None:
            print("no element to delete")
            return 
        if self.start_node.element == value: # if an element is found
            self.start_node = self.start_node.nref  # assign next reference of current node to start node
            return 
        n = self.start_node
        while n: # iterate until next references element
            if n.nref.element == value: 
                break
            n = n.nref
        if n.nref is None:
            print("item not found in index")
        else:
            n.nref = n.nref.nref

   
    def add_two_linked_lists(self,other):
        n1 = self.start_node 
        n2 = other.start_node
        l3 = SinglyLinkedList()
        l3.insert_at_start(0)
        n3 = l3.start_node

        while n1:
            l3.insert_at_end(n1.element+n2.element)
            n1 = n1.nref
            n2 = n2.nref 
        return l3

class DoublyLinkedList:
    """Create a new Doubly Linked List (dLinklist)
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
        self.size = 0

    def insert_in_empty_list(self, data):
        """inserts data into a empty dlinked list
        params: data to insert
        """
        if self.start_node is None:
            new_node = self.Node(data)
            self.start_node = new_node
            self.size += 1
        else:
            print("dlink is not empty")

    def insert_at_start(self, data):
        """inserets data at start of dlinked list
        takes O(1) time
        """
        if self.start_node is None: # if list is empty
            new_node = self.Node(data)
            self.start_node = new_node 
            self.size += 1
            return 
        new_node = self.Node(data) 
        new_node.nref = self.start_node  # make next reference of new node to current node
        self.start_node.pref = new_node # set previous reference of currenet node to new node
        self.start_node = new_node
        self.size += 1

    def insert_at_end(self,data):
        """inserts data at end of dlinked list"""
        if self.start_node is None:
            new_node = self.Node(data)
            self.start_node = new_node
            self.size += 1
            return
        n = self.start_node
        while n.nref:
            n = n.nref 
        new_node = self.Node(data)
        new_node.pref = n
        n.nref = new_node

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

    def insert_after_value(self, value, data):
        """inserts data after item x"""
        if self.start_node is None:
            print("list is empty")
            return 
        n = self.start_node
        while n:
            if n.element == value:
                break
            n = n.nref
        if n is None:
            print("value not found")
        else:
            new_node = self.Node(data)
            new_node.pref = n
            new_node.nref = n.nref
            if n.nref is not None: # if its not the last element
                n.nref.pref = new_node 
            n.nref = new_node

    def insert_before_value(self,value,data):
        if self.start_node is None:
            print("empty dlinked list")
            return
        if self.start_node.element == value:
            new_node = self.Node(data)
            new_node.nref = self.start_node
            self.start_node = new_node
            return 
        n = self.start_node
        while n:
            if n.element == x:
                break
            n = n.nref
        if n is None:
            print("dlinked list is empty")
        else:
            new_node = self.Node(data)
            new_node.nref = n  # set next ref of new to current node
            new_node.pref = n.pref # set previous ref of new node to prev ref of curr node
            n.pref = new_node # update pref reference of current node to new node

    def delete_from_start(self):
        """deletes a node from the start of dlinklist
            takes O(1) time
        """
        if self.start_node is None:
            print("dLinklist is empty")
            return
        if self.start_node.nref is None:
            self.start_node = None  # if only one element, delete this
            return
        self.start_node = self.start_node.nref
        self.start_node.pref = None

    def delete_from_end(self):
        """Deletes the node at end of dlist
            takes O(n) time
        """
        if self.start_node is None:
            print("dLinklist is empty")
            return
        n = self.start_node
        # iterate until next referene is not null
        while n.nref:
            n = n.nref
        n.pref.nref = None

    def delete_by_value(self, value):
        if self.start_node is None:
            print("dLinklist is empty")
            return
        # one element case
        if self.start_node.nref is None:
            if self.start_node.element == value:
                self.start_node = None
            else:
                print("Value not found")
            return
        # assumes value is found in first element, multiple node case
        if self.start_node.element == value:
            self.start_node = self.start_node.nref
            self.start_node.pref = None
            return
        # iterate until found
        n = self.start_node
        while n:
            if n.element == value:
                break
            n = n.nref
        if n is None:
            print("value not found")
        else:
            if n.nref is None: # last element case
                n.pref.nref = None
            else: # in the middle
                n.pref.nref = n.nref  # set the next reference of previous node to current nodes next reference
                n.nref.pref = n.pref # set the prev reference of next node to current nodes previous reference
    
    def reverse(self):
        if self.start_node is None:
            print("list has no elements to reverse")
            return 
        n = self.start_node # current node 
        prev_node = None 
        while n: # assume first node case...
            prev_node = n.pref  # save the previous reference 
            n.pref = n.nref # next ref of current node is prev ref of curr node after flip
            n.nref = prev_node
            n = n.pref # since pref is the next reference.  we iterate backward
        self.start_node = prev_node.pref

from abc import ABCMeta
class DoublyLinkedBase(metaclass=ABCMeta):
    """Abstract Base class for a doubly linked list

    Attributes:
    start_node:  (Node) represents head of the dlinked list
                        default set to None since its empty at time of creation
    end_node:    (Node) represents tail of the linked list
                        default set to None since its empty at time of creation"""
    class _Node:
        """
        Nested Node Class.
            Create a node to store value and 2 references for DoublyLinkedList
            Takes O(1) time

            params:
            element: represents element of the node
            pref:    previous reference of the node
            nref:    next reference of the node

        """

        def __init__(self, data, prev_ref=None, next_ref=None):
            self._element = data
            self._nref = next_ref
            self._pref = prev_ref

    def __init__(self):
        self._start_node = self._Node(None)
        self._end_node = self._Node(None)
        self._start_node._nref = self._end_node  # creating circular reference
        self._end_node._pref = self._start_node 
        self._size = 0

    def __len__(self):
        return self._size 

    def is_empty(self):
        """Returns True if dlinklist is empty"""
        return self.size == 0

    def _insert_between(self, data, node1, node2):
        """Adds data between two nodes"""
        new_node = self._Node(data, node1, node2)
        node1._nref = new_node
        node2._pref = new_node
        self._size += 1
        return new_node

    def _delete_node(self,node):
        """delete node from list and return the element"""
        before = node._pref 
        after = node._nref 
        before._nref = after 
        after._pref = before 
        self._size -= 1
        element = node._element 
        node._nref = node._pref = node._element = None   # deprecate Node
        return element 

    def _traverse(self):
        """Traverses a Linked List
            Takes O(n) time
        """
        if self._start_node is None:
            print("list has no elements")
        else:
            n = self._start_node
            while n is not None:
                print(n._element, " ")
                n = n._nref
