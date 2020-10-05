"""Linear Collections, consisting of stacks, queue and deque"""

from typing import Any, Union, List
from collections import namedtuple

class _Empty(Exception):
    """prints Exception error if array is empty

    :param Exception: message for exception error
    :type Exception: string
    :return: Exception Message
    :rtype: Exception Error
    """
    pass

class Stack:
    """Implementation of stacks using lists
    
       O(1) time for all operations
       O(n) time for max operation
       Assumes LIFO; 
       items on the right is the top, left is bottom
    """
    def __init__(self, data: Union[None, List]):
        self._data = [] if data is None else data
    
    def empty(self):
        """returns true if stack is empty"""
        return self._data == []
    
    def pop(self):
        """removes item from right(top) of stack"""
        if self.empty():
            raise _Empty("Stack is empty")
        return self._data.pop()
    
    def push(self, item: Any) -> None:
        """adds item to right(top) of stack

        :param item: item to be pushed
        :type item: Any
        """
        self._data.append(item)
    
    def peek(self) -> Any:
        """returns right(top) item of stack

        :return: [description]
        :rtype: Any
        """
        return self._data[-1]
    
    def size(self) -> int:
        """returns number of items in stack

        :return: length of stack
        :rtype: int
        """
        return len(self._data)
    
    def max(self) -> Union[int, float]:
        """returns the maximum item in stack

        Takes O(n) worse time

        :return: [description]
        :rtype: Any[int, float]
        """
        return max(self._data)


class MaxStack:
    """Implementation of stacks using lists
    
       O(1) time for all operations
       Assumes LIFO; 
       items on the right is the top, left is bottom
    """
    # to keep track of maximum value after each push
    _Items = namedtuple("_Items", ('item', 'max'))

    def __init__(self, item: Union[None, List]):
        self._data = []
        self._create_stack(item)
        
    def _create_stack(self, items: list) -> list:
        """Creates a stack from list of items

        :param items: list of items
        :type items: list
        :return: list that follows a stack order
        :rtype: list
        """
        # push items to a stack if user supplies list of items
        if isinstance(items, list):
            for item in items:
                self.push(item)

    def empty(self):
        """returns True if Stack is empty

        :return: True/False
        :rtype: bool
        """
        return self._data == []
    
    def max(self) -> Union[float, int]:
        """returns maximum value in a stack

        Takes O(1) time

        :raises _Empty: if stack is empty, raise Exception
        :return: maximum value in a stack
        :rtype: Union[float, int]
        """
        if self.empty():
            raise _Empty("Stack is empty")
        return self._data[-1].max
    
    def pop(self) -> Any:
        if self.empty():
            raise _Empty("Stack is empty")
        return self._data[-1].item
    
    def push(self, item):
        self._data.append(\
            self._Items(item, item
            if self.empty() else max(item, self.max())))

class Queue: