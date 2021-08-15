# pyre-strict
"""Linear Collections, consisting of stacks, queue and deque"""
import collections
from dataclasses import dataclass
from typing import Any, List, Optional, Union


class Stack:
    """Implementation of stacks using lists

    O(1) time for all operations
    O(n) time for max operation
    Assumes LIFO;
    items on the right is the top, left is bottom
    """

    def __init__(self, data: Optional[list] = None):
        self._data = [] if data is None else data

    def empty(self) -> bool:
        """returns true if stack is empty"""
        return self._data == []

    # pyre-ignore
    def pop(self) -> Any:
        """removes item from right(top) of stack"""
        if self.empty():
            raise _Empty("Stack is empty")
        return self._data.pop()

    # pyre-ignore
    def push(self, item: Any) -> None:
        """adds item to right(top) of stack

        :param item: item to be pushed
        :type item: Any
        """
        self._data.append(item)

    # pyre-ignore
    def peek(self) -> Any:
        """returns right(top) item of stack

        :return: [description]
        :rtype: Any
        """
        return self._data[-1]

    def size(self) -> int:
        """returns number of items in a stack

        Returns
        -------
        int
            the length of stack
        """
        return len(self._data)

    def max(self) -> Union[int, float]:
        """returns the maximum item in stack

        Takes O(n) worse time

        :return: [description]
        :rtype: Any[int, float]
        """
        return max(self._data)


@dataclass
class Item:
    item: Any  # pyre-ignore
    max: Any  # pyre-ignore


@dataclass
class MaxStack:
    """Implementation of stacks using lists

    O(1) time for all operations
    Assumes LIFO;
    items on the right is the top, left is bottom
    """

    _data: List[Item] = []

    # pyre-ignore
    def _create_stack(self, items: List[Any]) -> None:
        """Creates a stack from list of items

        Parameters
        ----------
        items : list
            [description]
        """
        # push items to a stack if user supplies list of items
        for item in items:
            self.push(item)

    def is_empty(self) -> bool:
        return self._data == []

    def max(self) -> Union[float, int]:
        """Returns maximum value in a stack

        Returns
        -------
        Union[float, int]
            [description]

        Raises
        ------
        _Empty
            [description]
        """
        if self.is_empty():
            raise _Empty("Stack is empty")
        return self._data[-1].max

    # pyre-ignore
    def pop(self) -> Any:
        """removes item from the right(top) of stack

        Returns
        -------
        Any
            [description]

        Raises
        ------
        _Empty
            [description]
        """
        if self.is_empty():
            raise _Empty("Stack is empty")
        return self._data[-1].item

    # pyre-ignore
    def push(self, item: Any) -> None:
        """adds item to right(top) of stack

        :param item: item to be pushed
        :type item: Any
        """
        self._data.append(
            Item(item, item if self.is_empty() else max(item, self.max()))
        )


class Queue:
    """Implementation of Queue using a list

    O(1) for enqueue operation
    O(n) for dequeue operation
    Assumes: First In First Out (FIFO)
    """

    def __init__(self) -> None:
        """Default constructor, needs no parameters"""
        self._data: List[Any] = []  # pyre-ignore

    def empty(self) -> bool:
        return self._data == []

    # pyre-ignore
    def enqueue(self, item: Any) -> None:
        self._data.append(item)

    # pyre-ignore
    def dequeue(self) -> Any:
        return self._data.pop(0)

    def size(self) -> int:
        return len(self._data)

    def max(self) -> Union[float, int]:
        return max(self._data)


class CircularQueue:
    """Implementation of a Queue using a circular Array

    O(1) time for all operations
    Assumes: First In First Out (FIFO)
    """

    # class constant
    _DEFAULT_CAPACITY = 10

    def __init__(self):
        """Default Constructor, needs no parameters"""
        self._data = [None] * CircularQueue._DEFAULT_CAPACITY
        self._size = 0
        self._front = 0

    def __len__(self):
        return self._size

    def size(self):
        return len(self)

    def empty(self):
        return self.size() == 0

    def front(self):
        """Returns, but doesn't remove first element"""
        if self.empty():
            raise IndexError("Queue is Empty")
        return self._data[self._front]

    def dequeue(self):
        if self.empty():
            raise IndexError("Queue is Empty")
        removed_item = self.front()
        # reclaim for garbage collection
        self._data[self._front] = None
        # get available space and decrement size by 1
        self._front = (1 + self._front) % len(self._data)
        self._size -= 1
        return removed_item

    def enqueue(self, item):
        # if capcacity is reached, double capacity
        if self.size() == len(self._data):
            self._resize(2 * len(self._data))
        avail = (self._front + self.size()) % len(self._data)
        self._data[avail] = item
        self._size += 1

    def _resize(self, cap):
        old = self._data
        self._data = [None] * cap
        walk = self._front
        for k in range(self.size()):
            self._data[k] = old[walk]
            walk = (1 + walk) % len(old)
        self._front = 0


class StackQueue:
    """Implementation of Queue using Stacks"""

    def __init__(self):
        self._enq, self._deq = Stack(), Stack()

    def enqueue(self, item):
        self._enq.push(item)

    def dequeue(self):
        if not self._deq:
            while self._enq:
                self._deq.push(self._enq.pop())
        if not self._deq:
            raise IndexError("Empty Queue")
        return self._deq.pop()

    def front(self):
        return self._enq.peek()


class MaxQueue:
    """Implementation of Queue withh max API using Deque"""

    def __init__(self):
        self._data = collections.deque()
        self._candidates_for_max = collections.deque()

    def enqueue(self, item):
        pass
