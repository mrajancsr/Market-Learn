# pyre-strict
"""Implementation of a Doubly Linked List"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from marketlearn.algorithms.linked_lists.linked_base import (
    EmptyException,
    Node,
)


@dataclass
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

    start_node: Optional[Node] = field(init=False, default=None)
    size: int = field(init=False, default=0)

    # pyre-ignore
    def insert_in_empty_list(self, data: Any) -> None:
        """inserts data into a empty dlinked list
        params: data to insert
        """
        if self.start_node is None:
            new_node = Node(data)
            self.start_node = new_node
            self.size += 1
        else:
            raise EmptyException("Doubly Linked List Cannot be Empty")

    # pyre-ignore
    def insert_at_start(self, data: Any) -> None:
        """inserets data at start of dlinked list
        takes O(1) time
        """
        if self.start_node is None:
            new_node = Node(data)
            self.start_node = new_node
            self.size += 1
            return None

        new_node = Node(data)
        # make next reference of new node to current node
        new_node.nref = self.start_node

        # set previous reference of currenet node to new node
        self.start_node.pref = new_node
        self.start_node = new_node
        self.size += 1

    # pyre-ignore
    def insert_at_end(self, data) -> None:
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

    def traverse(self) -> None:
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

    # pyre-ignore
    def insert_after_value(self, value: Any, data: Any) -> None:
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
            new_node = Node(data)
            new_node.pref = n
            new_node.nref = n.nref

            # if its not the last element
            if n.nref is not None:
                n.nref.pref = new_node
            n.nref = new_node

    # pyre-ignore
    def insert_before_value(self, value: Any, data: Any) -> None:
        if self.start_node is None:
            print("empty dlinked list")
            return
        if self.start_node.element == value:
            new_node = Node(data)
            new_node.nref = self.start_node
            self.start_node = new_node
            return
        n = self.start_node
        while n:
            if n.element == data:
                break
            n = n.nref
        if n is None:
            print("dlinked list is empty")
        else:
            new_node = Node(data)
            # set next ref of new to current node
            new_node.nref = n

            # set previous ref of new node to prev ref of curr node
            new_node.pref = n.pref

            # update pref reference of current node to new node
            n.pref = new_node

    def delete_from_start(self) -> None:
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

    def delete_from_end(self) -> None:
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

    def delete_by_value(self, value) -> None:
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

    def reverse(self) -> None:
        if self.start_node is None:
            print("list has no elements to reverse")
            return
        n = self.start_node  # current node
        prev_node = None
        # assume first node case...
        while n:
            # save the previous reference
            prev_node = n.pref
            # next ref of current node is prev ref of curr node after flip
            n.pref = n.nref
            n.nref = prev_node
            # since pref is the next reference.  we iterate backward
            n = n.pref
        self.start_node = prev_node.pref
