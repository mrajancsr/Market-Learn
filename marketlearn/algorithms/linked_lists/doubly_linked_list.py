"""Implementation of a Doubly Linked List"""

from typing import Any


class DoublyLinkedList:
    """Create a new Doubly Linked List (dLinklist)
        Takes O(1) time

    Parameters
    ----------
    None

    Attributes
    ----------
    start_node: Node, default=None
        represents head of the linked list
        default set to None since its empty at time of creation

    size: int
        represents the length of the linked list

    """

    class Node:
        """Nested Node Class
        Create a node to store v alue and 2 references for LinkedList
        Takes O(1) time

        Parameters
        ----------
        data : Any, default=None
            represents element of the node
        next_ref : optional[self.Node], default=None
            next referernce of the node
        prev_ref : optional[self.Node], default=None
            previous reference of the node
        """

        def __init__(self, data: Any, next_ref: "Node" = None, prev_ref: "Node" = None):
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
        if self.start_node is None:  # if list is empty
            new_node = self.Node(data)
            self.start_node = new_node
            self.size += 1
            return
        new_node = self.Node(data)
        # make next reference of new node to current node
        new_node.nref = self.start_node

        # set previous reference of currenet node to new node
        self.start_node.pref = new_node
        self.start_node = new_node
        self.size += 1

    def insert_at_end(self, data):
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

            # if its not the last element
            if n.nref is not None:
                n.nref.pref = new_node
            n.nref = new_node

    def insert_before_value(self, value, data):
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
            # set next ref of new to current node
            new_node.nref = n

            # set previous ref of new node to prev ref of curr node
            new_node.pref = n.pref

            # update pref reference of current node to new node
            n.pref = new_node

    def delete_from_start(self):
        """deletes a node from the start of dlinklist
        takes O(1) time
        """
        if self.start_node is None:
            print("dLinklist is empty")
            return
        if self.start_node.nref is None:
            # if only one element, delete this
            self.start_node = None
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
            # last element case
            if n.nref is None:
                n.pref.nref = None

            # in the middle
            else:
                # set the next ref of prev node to current nodes next ref
                n.pref.nref = n.nref

                # set the prev ref of next node to current nodes previous ref
                n.nref.pref = n.pref

    def reverse(self):
        if self.start_node is None:
            print("list has no elements to reverse")
            return
        n = self.start_node  # current node
        prev_node = None
        while n:  # assume first node case...
            prev_node = n.pref  # save the previous reference
            n.pref = (
                n.nref
            )  # next ref of current node is prev ref of curr node after flip
            n.nref = prev_node
            n = n.pref  # since pref is the next reference.  we iterate backward
        self.start_node = prev_node.pref