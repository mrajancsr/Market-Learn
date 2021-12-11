# pyre-strict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from marketlearn.algorithms.linked_collections import Position, PositionalList
from marketlearn.algorithms.linked_lists.linked_base import EmptyException
from marketlearn.algorithms.priority_queues.queue_base import (
    Item,
    PriorityQueueBase,
)


@dataclass
class UnsortedPriorityQueue(PriorityQueueBase):
    _data: PositionalList = field(init=False, default=PositionalList())

    def __len__(self) -> int:
        return len(self._data)

    # pyre-ignore
    def add(self, key: Any, value: Any) -> None:
        self._data.add_last(Item(key, value))

    def _find_min(self) -> Optional[Position]:
        if self.is_empty():
            raise EmptyException("Priority Queue is Empty")
        small = self._data.first()
        if small:
            walk = self._data.after(small)
            while walk is not None:
                if walk.element() < small.element():
                    small = walk
                walk = self._data.after(walk)
            return small

    # pyre-ignore
    def min(self) -> Optional[Tuple[Any, Any]]:
        p = self._find_min()
        if p:
            element: Item = p.element()
            return (element.key, element.value)

    # pyre-ignore
    def remove_min(self) -> Optional[Tuple[Any, Any]]:
        p = self._find_min()
        if p:
            item: Item = self._data.delete(p)
            return (item.key, item.value)


@dataclass
class SortedPriorityQueue(PriorityQueueBase):
    _data: PositionalList = field(init=False, default=PositionalList())

    def __len__(self) -> int:
        return len(self._data)

    # pyre-ignore`
    def add(self, key: Any, value: Any) -> None:
        """adds an item

        Parameters
        ----------
        key : Any
            [description]
        value : Any
            [description]
        """
        latest = Item(key, value)
        # assume last element is largest
        walk = self._data.last()
        while walk is not None and latest < walk.element():
            walk = self._data.before(walk)
        if walk is None:
            self._data.add_first(latest)
        # walk is less than latest so add after walk
        else:
            self._data.add_after(latest, walk)

    # pyre-ignore
    def min(self) -> Optional[Tuple[Any, Any]]:
        if self.is_empty():
            raise EmptyException("Priority Queue is Empty")
        p = self._data.first()
        if p:
            item: Item = p.element()
            return (item.key, item.value)

    # pyre-ignore
    def remove_min(self) -> Optional[Tuple[Any, Any]]:
        if self.is_empty():
            raise EmptyException("Priority Queue is Empty")
        p = self._data.first()
        if p:
            item: Item = self._data.delete(p)
            return (item.key, item.value)


@dataclass
class HeapPriorityQueue(PriorityQueueBase):
    """Array based representation of PriorityQueue"""

    _data: List[Any] = field(init=False, default_factory=list)  # pyre-ignore

    def __len__(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return len(self) == 0

    def get_parent_index(self, child_index: int) -> int:
        """Returns index of parent of child index

        Parameters
        ----------
        child_index : int
            index of child position

        Returns
        -------
        int
            index of parent of child index j
        """
        return (child_index - 1) // 2

    def get_left_child_index(self, parent_index: int) -> int:
        """Returns index of left child of parent index

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        int
            index of left child of parent_index
        """
        return 2 * parent_index + 1

    def get_right_child_index(self, parent_index: int) -> int:
        """Returns index of right child of parent index

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        int
            index of right child of parent_index
        """
        return 2 * parent_index + 2

    def has_left_child(self, parent_index: int) -> bool:
        """Returns true if parent index has a left child

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        bool
            true if parent has a left child
        """
        index_of_left_child = self.get_left_child_index(parent_index)
        return index_of_left_child < len(self)

    def has_right_child(self, parent_index: int) -> bool:
        """Returns True if parent index has a right child

        Parameters
        ----------
        parent_index : int
            index of parent

        Returns
        -------
        bool
            true if parent has a right child
        """
        index_of_right_child = self.get_right_child_index(parent_index)
        return index_of_right_child < len(self)

    def swap_entries(self, i: int, j: int) -> None:
        """Swaps entries at position i and j

        Parameters
        ----------
        i : int
            position to be swapped
        j : int
            position to be swapped
        """
        self._data[i], self._data[j] = self._data[j], self._data[i]
        self._data[i].index, self._data[j].index = i, j

    def upheap_bubbling(self, j: int) -> None:
        """Performs upheap bubbling via recrusion
        T(n) ~ O(h) where h is height of tree
        and h = log(n)

        Parameters
        ----------
        j : int
            index of item inserted into the queue
        """
        parent_index = self.get_parent_index(j)
        if j > 0 and self._data[parent_index] > self._data[j]:
            self.swap_entries(j, parent_index)
            self.upheap_bubbling(parent_index)

    # pyre-ignore
    def push(self, key: Any, value: float) -> Item:
        """Adds item to the priority queue and performs upheap bubbling after insertion

        Parameters
        ----------
        key : Any
            [description]
        value : Any
            [description]
        """
        item = Item(key, value)
        self._data.append(item)
        start_index = len(self) - 1
        self.upheap_bubbling(start_index)
        return item

    def downheap_bubbling(self, j: int) -> None:
        """Pergforms downheap bubbling via recursion

        Parameters
        ----------
        j : int
            [description]
        """
        left_child_index = None
        right_child_index = None
        small_child_index = None
        if self.has_left_child(j):
            left_child_index = self.get_left_child_index(j)
            small_child_index = left_child_index
        if self.has_right_child(j):
            right_child_index = self.get_right_child_index(j)
        if left_child_index and right_child_index:
            if self._data[right_child_index] <= self._data[left_child_index]:
                small_child_index = right_child_index
        if small_child_index:
            if self._data[small_child_index] <= self._data[j]:
                self.swap_entries(j, small_child_index)
                self.downheap_bubbling(small_child_index)

    # pyre-ignore
    def pop(self) -> Tuple[Any, Any]:
        if self.is_empty():
            raise EmptyException("PQ is empty")
        n = len(self) - 1
        # put minimum item at end, remove it
        self.swap_entries(0, n)
        item: Item = self._data.pop()
        # perform downheap bubbling starting from root
        self.downheap_bubbling(0)
        return (item.key, item.value)
