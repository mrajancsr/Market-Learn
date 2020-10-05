"""Linear Collections, implementation of stack, queue and deque
   using lists
"""

class Stack:
    """
        O(1) time for all operations
        Assumes items on the right
        is the top
    """

    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def pop(self):
        self.items.pop()

    def push(self, item):
        self.items.append(item)

    def peek(self):
        return self.items[-1]

    def __repr__(self):
        return str(self.items)


class Queue:
    """Assumes top is the left
        and end is right
        O(n) time for insertion
        O(1) time for removal
    """
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def size(self):
        return len(self.items)

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def __repr__(self):
        return repr(self.items)


class Deque:
    """Assumes front is left,
        rear is right
        front time in O(n)
        rear time in O(1)
    """
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def add_front(self, item):
        self.items.insert(0, item)

    def add_rear(self, item):
        self.items.append(item)

    def remove_front(self):
        return self.items.pop(0)

    def remove_rear(self):
        return self.items.pop()

    def __repr__(self):
        return repr(self.items)

    def size(self):
        return len(self.items)

