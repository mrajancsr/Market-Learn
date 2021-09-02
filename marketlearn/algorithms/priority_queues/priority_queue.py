# pyre-strict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from marketlearn.algorithms.linked_collections import Position, PositionalList
from marketlearn.algorithms.linked_lists.linked_base import EmptyException
from marketlearn.algorithms.priority_queues import Item, PriorityQueueBase


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
    _data: List[Any] = field(init=False, default_factory=list)  # pyre-ignore

    def __len__(self) -> int:
        return len(self._data)

    def _parent(self, j: int) -> int:
        """Returns index of parent of child index j

        Parameters
        ----------
        j : int
            index of child position

        Returns
        -------
        int
            index of parent of child index j
        """
        return (j - 1) // 2

    def _left(self, j: int) -> int:
        return 2 * j + 1

    def _right(self, j: int) -> int:
        return 2 * j + 2

    def _has_left(self, j: int) -> bool:
        return self._left(j) < len(self)

    def _has_right(self, j: int) -> bool:
        return self._right(j) < len(self)

    def _swap(self, i: int, j: int) -> None:
        self._data[i], self._data[j] = self._data[j], self._data[i]

    def _upheap(self, j: int) -> None:
        parent = self._parent(j)
        if j > 0 and self._data[j] < self._data[parent]:
            self._swap(j, parent)
            self._upheap(parent)

    # pyre-ignore
    def add(self, key: Any, value: Any) -> None:
        self._data.append(Item(key, value))
        self._upheap(len(self) - 1)

    # pyre-ignore
    def min(self) -> Optional[Tuple[Any, Any]]:
        if self.is_empty():
            raise EmptyException("Queue is Empty")
        item: Item = self._data[0]
        return (item.key, item.value)

    def _downheap(self, j: int) -> None:
        pass
