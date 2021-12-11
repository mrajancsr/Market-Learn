"""linear collections of Stack, Queue and Deque implemented via linked List
Takes O(1) time for all insertion, removal operations
-O(n) time complexity for traversing the ADTs to print the elements
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from marketlearn.algorithms.linked_lists.linked_base import (
    DoublyLinkedBase,
    EmptyException,
    Node,
)
from marketlearn.algorithms.linked_lists.linked_base import (
    Position as PositionBase,
)


@dataclass
class Position(PositionBase):
    """Class represents position of single element"""

    _container: object
    _node: Node

    # pyre-ignore
    def element(self) -> Any:
        """Returns element stored at this position

        Returns
        -------
        [Any]
            Returns element stored at this position
        """
        return self._node.element

    def __eq__(self, other: Position) -> bool:
        """Return true if other posotion represents same location

        Parameters
        ----------
        other : Position
            [description]
        """
        return type(other) is type(self) and other._node is self._node

    def __ne__(self, other: Position) -> bool:
        return not (self == other)


@dataclass
class LinkedStack:
    """LIFO Stack Implementation using a singly linked list
        Takes O(1) time

    params:
    None

    Attributes:
    start_node:  (Node) represents head of the linked list
                        default set to None since its empty at time of creation
    """

    start_node: Optional[Node] = None
    size: int = 0

    def __len__(self) -> int:
        """Returns number of elements in a stack
        takes O(1) time"""
        return self.size

    def is_empty(self) -> bool:
        """Return True if Stack is empty
        takes O(1) time"""
        return self.size == 0

    # pyre-ignore
    def pop(self) -> Any:
        """Removes and returns element from top of the stack (LIFO)
        takes O(1) time
        """
        if self.is_empty():
            raise EmptyException("Stack has no elements to delete")
        if self.start_node:
            element = getattr(self.start_node, "element")
            nref = getattr(self.start_node, "nref")
            # delete by assigning next reference of start node to start node
            self.start_node = nref
            self.size -= 1
            return element

    def push(self, data: Any) -> None:
        """Adds element to top of the Stack
        Takes O(1) time
        """
        new_node = Node(data)  # create a node
        new_node.nref = (
            self.start_node
        )  # new nodes next ref is old nodes start
        self.start_node = new_node  # new node is now the start_node
        self.size += 1

    def peek(self) -> Any:
        """Returns but does not remove element from top of Stack
        takes O(1) time
        """
        if self.is_empty():
            raise EmptyException("stack is empty")
        if self.start_node is not None:
            element = self.start_node.element
            return element
        return None

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


@dataclass
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

    start_node: Optional[Node] = field(init=False, default=None)
    end_node: Optional[Node] = field(init=False, default=None)
    size: int = field(init=False, default=0)

    def __len__(self) -> int:
        """Returns the number of items in a Queue
        takes O(1) time
        """
        return self.size

    def is_empty(self) -> bool:
        """Returns True if Queue is empty
        takes O(1) time
        """
        return self.size == 0

    # pyre-ignore
    def enqueue(self, data: Any) -> None:
        """Adds element to back of the Queue
        Takes O(1) time
        """
        new_node = Node(data)  # create a node
        if self.is_empty():
            self.start_node = new_node
            self.end_node = new_node  # creating a circular reference
            self.size += 1
            return None
        # else set the next reference of end node
        nref = getattr(self.end_node, "nref")
        nref = new_node  # noqa
        self.end_node = new_node  # update ref of end node to new node
        self.size += 1

    # pyre-ignore
    def dequeue(self) -> Any:
        """removes item at top of the Queue
        takes O(1) time
        """
        if self.is_empty():
            raise EmptyException("Queue is Empty")
        element = getattr(self.start_node, "element")

        # delete by assigning next reference of start node to start node
        self.start_node = getattr(self.start_node, "nref")
        self.size -= 1

        # special case for one element
        if self.is_empty():
            # enforce tail node to None
            self.end_node = None
        return element

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


@dataclass
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

    Todo: Need to test the class for correctness

    """

    start_node: Optional[Node] = None
    end_node: Optional[Node] = None
    size: int = 0

    def __len__(self) -> int:
        return self.size

    def is_empty(self) -> bool:
        return self.size == 0

    # pyre-ignore
    def add_front(self, data: Any) -> None:
        """inserets data at start of Deque
        takes O(1) time
        """
        if self.is_empty():
            new_node = Node(data)
            self.start_node = new_node
            self.end_node = new_node  # creating a circular reference
            self.size += 1
            return
        new_node = Node(data)
        # make next reference of new node to current node
        new_node.nref = self.start_node

        # set previous reference of currenet node to new node
        pref = getattr(self.start_node, "pref")
        pref = new_node  # noqa
        self.start_node = new_node
        self.size += 1

    # pyre-ignore
    def remove_front(self) -> Any:
        """deletes a node from the start of Deque
        and returns the element
        takes O(1) time
        """
        if self.is_empty():
            raise EmptyException("Queue is Empty")
        element = getattr(self.start_node, "element")
        nref = getattr(self.start_node, "nref")
        if nref is None:
            self.start_node = None  # if only one element, delete this
            self.size -= 1
            return element
        self.start_node = nref
        pref = getattr(self.start_node, "pref")
        pref = None  # noqa
        self.size -= 1
        # one element case after deleting previous node
        if self.is_empty():
            self.end_node = None
        return element

    # pyre-ignore
    def add_rear(self, data: Any) -> None:
        """Adds element to back of the Deck
        Takes O(1) time
        """
        new_node = Node(data)  # create a node
        if self.is_empty():
            self.start_node = new_node
            self.end_node = new_node  # creating a circular reference
            self.size += 1
            return
        # else set the next reference of end node
        nref = getattr(self.end_node, "nref")
        nref = new_node  # noqa
        new_node.pref = self.end_node
        self.end_node = new_node  # update ref of end node to new node
        self.size += 1

    # pyre-ignore
    def remove_rear(self) -> Any:
        """removes element at back of the Deck
        Takes O(1) time
        """
        # to do.  not correctly removing rear
        if self.is_empty():
            raise EmptyException("Deque is Empty")

        element = getattr(self.end_node, "element")

        self.end_node = getattr(self.end_node, "pref")
        nref = getattr(self.end_node, "nref")
        nref = None  # noqa
        if self.is_empty():
            self.end_node = None
        self.size -= 1
        return element

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


class PositionalList(DoublyLinkedBase):
    """Container of elements allowing positional access"""

    # - Utility methods
    def _validate(self, p: Position) -> Node:
        """Return position's node, or raise error if invalid

        Parameters
        ----------
        p : Position
            [description]
        """
        if not isinstance(p, Position):
            raise TypeError("p must be proper Position type")
        if p._container is not self:
            raise ValueError("p does not belong to this container")
        if p._node.nref is None:
            raise ValueError("p is no longer valid")
        return p._node

    def _make_position(self, node: Node) -> Optional[Position]:
        """Return position instance for given node or None if sentinel"""
        if node is self.start_node or node is self.end_node:
            return None
        return Position(self, node)

    # - Accessors
    def first(self) -> Optional[Position]:
        """Return first position in list or None if empty

        Returns
        -------
        Union[Position, None]
            first position in list
        """
        if self.is_empty():
            return None
        nref = getattr(self.start_node, "nref")
        return self._make_position(nref)

    def last(self) -> Optional[Position]:
        """Return last position in list or None if empty

        Returns
        -------
        Union[Position, None]
            last position in list
        """
        pref = getattr(self.end_node, "pref")
        return self._make_position(pref)

    def before(self, p: Position) -> Optional[Position]:
        """Return position before p's position or None if p is first psotion

        Parameters
        ----------
        p : Position
            [description]

        Returns
        -------
        Union[Position, None]
            [description]
        """
        node = self._validate(p)
        pref = getattr(node, "pref")
        return self._make_position(pref)

    def after(self, p: Position) -> Optional[Position]:
        """Return position after p's position or None if p is last position

        Parameters
        ----------
        p : [type]
            [description]

        Returns
        -------
        Union[Position, None]
            [description]
        """
        node = self._validate(p)
        nref = getattr(node, "nref")
        return self._make_position(nref)

    # pyre-ignore
    def __iter__(self) -> Iterator[Any]:
        """Generate forward iteration of elements in list"""
        cursor = self.first()
        while cursor is not None:
            yield cursor.element()
            cursor = self.after(cursor)

    def insert_between(
        self,
        data: Any,  # pyre-ignore
        node1: Node,
        node2: Node,
    ) -> Optional[Position]:
        """Inserts data between two nodes and returns position object

        Parameters
        ----------
        data : Any
            [description]
        node1 : Node
            node after which data is inserted
        node2 : Node
            node before which data is inserted

        Returns
        -------
        Position
            of data between node1 and node2
        """
        node = self._insert_between(data, node1, node2)
        if node:
            return self._make_position(node)

    # pyre-ignore
    def add_first(self, data: Any) -> Optional[Position]:
        """Insert data into first position and return new position

        Parameters
        ----------
        data : Any
            [description]

        Returns
        -------
        Position
            of data inserted
        """
        nref = getattr(self.start_node, "nref")
        return self.insert_between(data, self.start_node, nref)

    # pyre-ignore
    def add_last(self, data: Any) -> Optional[Position]:
        """Insert data into last position and return new position

        Parameters
        ----------
        data : Any
            data to be inserted into the list

        Returns
        -------
        Position
            data inserted as last item
        """
        pref = getattr(self.end_node, "pref")
        return self.insert_between(data, pref, self.end_node)

    # pyre-ignore
    def add_before(self, data: Any, p: Position) -> Optional[Position]:
        """Insert data right before position p and return new position

        Parameters
        ----------
        data : Any
            data to be inserted in list
        p : Position
            data is inserted before this position

        Returns
        -------
        Position
            position of inserted object
        """
        node = self._validate(p)
        pref = getattr(node, "pref")
        return self.insert_between(data, pref, node)

    # pyre-ignore
    def add_after(self, data: Any, p: Position) -> Optional[Position]:
        node = self._validate(p)
        nref = getattr(node, "nref")
        return self.insert_between(data, node, nref)

    # pyre-ignore
    def delete(self, p: Position) -> Any:
        """Remove and return the element at postion p

        Parameters
        ----------
        p : [type]
            [description]
        """
        node = self._validate(p)
        return self._delete_node(node)

    # pyre-ignore
    def replace(self, p: Position, data: Any) -> Any:
        """Replace

        Parameters
        ----------
        p : Position
            [description]
        data : Any
            [description]

        Returns
        -------
        Any
            [description]
        """
        original_node = self._validate(p)
        old_value = original_node.element
        original_node.element = data
        return old_value
