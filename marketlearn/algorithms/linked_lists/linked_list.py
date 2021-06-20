"""Implementation for Singly and DoublyLinked List
Author: Rajan Subramanian
"""

from __future__ import annotations
from typing import Any


class SinglyLinkedList:
    """Create a Singly Linked List
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

        Create a node to store value and reference for LinkedList
        Takes O(1) time

        Parameters
        ----------
        data : Any
            represents element stored in a node
        reference : Node, optional, default=None
            next reference of the Node
        """

        def __init__(self, data: Any, reference: Node = None):
            self.element = data
            self.nref = reference

    def __init__(self):
        self.start_node = None
        self.size = 0

    def __len__(self):
        """returns length of linked list

        Returns
        -------
        int
            returns length of linked list
        """
        return self.size

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

    def insert_at_start(self, data: Any):
        """inserts data at start of the list
        takes O(1) time

        Parameters
        ----------
        data : Any
            data to be inserted into linkedlist
        """
        new_node = self.Node(data)  # create a node
        new_node.nref = self.start_node  # new nodes nextref is old nodes start
        self.start_node = new_node  # new node is now the start_node
        self.size += 1

    def insert_at_end(self, data: Any) -> None:
        """inserts value at end of the list
        if list is empty, takes O(1) time.  Otherwise:
        O(n) time

        Parameters
        ----------
        data : Any
            data to be inserted into linkedlist
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

    def insert_after_item(self, item: Any, data: Any):
        """nserts data before item is found
        if found, takes O(k+1), otherwise, O(n)

        Parameters
        ----------
        item : Any
            element contained in Node
        data : Any
            the data to be inserted after item if found
        """
        n = self.start_node
        while n:  # iterate until x is found
            if n.element == item:
                break
            n = n.nref
        if n is None:
            print("item not found in list")
        else:
            new_node = self.Node(data)
            # set the new nodes next ref to current nodes next ref
            new_node.nref = n.nref
            n.nref = new_node  # set the current nodes next ref to new node
            self.size += 1

    def insert_before_item(self, item: Any, data: Any):
        """inserts data before item is found
        if found, takes O(k+1), otherwise, O(n)

        Parameters
        ----------
        item : Any
            element contained in node
        data : Any
            data to be inserted before item is found
        """
        if self.start_node is None:
            print("list has no elements")
            return

        # if element is found in first node
        if self.start_node.element == item:
            new_node = self.Node(data)
            new_node.nref = self.start_node
            self.start_node = new_node
            self.size += 1
            return

        # iterate until next node contains element
        n = self.start_node
        while n.nref:
            if n.nref.element == item:
                break
            n = n.nref

        # if end of list is reached
        if n.nref is None:
            print("value not found is list")
        else:
            new_node = self.Node(data)
            new_node.nref = n.nref  # new nodes next ref is previous nodes next
            n.nref = new_node
            self.size += 1

    def insert_at_index(self, index, data):
        if index == 0:
            new_node = self.Node(data)
            new_node.nref = self.start_node
            self.start_node = new_node
            self.size += 1
            return

        # iterate until index is found
        n = self.start_node
        while n and index > 1:
            index -= 1
            n = n.nref

        if n.nref is None:
            raise IndexError("Index out of bounds")
        else:
            new_node = self.Node(data)
            new_node.nref = n.nref
            n.nref = new_node
            self.size += 1

    def count_positions(self):
        """Returns the count of nodes in Linked List"""
        return self.size

    def search(self, item: Any) -> bool:
        """returns True if item is found

        Parameters
        ----------
        item : Any
            the item to look for

        Returns
        -------
        bool
            return True if item is found
        """
        # check if linked list is empty
        if self.start_node is None:
            print("list has non elements")
            return

        # iterate and search for elements
        n = self.start_node
        while n:
            if n.element == item:
                return True
            n = n.nref
        return False

    def reverse(self):
        n = self.start_node  # current node
        prev_node = None
        next_node = None
        while n:  # assume first node case...
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

    def delete_item(self, item: Any):
        """Deletes the Node that contains the item

        :param item: the item to look for and remove
        :type item: Any
        """
        # check if list is empty
        if self.start_node is None:
            print("no element to delete")
            return
        # check if first node contains the item
        if self.start_node.element == item:
            # assign next reference of current node to start node
            self.start_node = self.start_node.nref
            return

        # iterate until element is found
        n = self.start_node
        while n:
            if n.nref.element == item:
                break
            n = n.nref

        # check if tail is reached
        if n.nref is None:
            print("item not found in index")
        else:
            n.nref = n.nref.nref