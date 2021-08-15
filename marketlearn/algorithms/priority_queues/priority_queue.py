# pyre-strict
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

from marketlearn.algorithms.linked_collections import Position, PositionalList
from marketlearn.algorithms.priority_queues import Item, PriorityQueueBase


class Empty(Exception):
    pass


@dataclass
class UnsortedPriorityQueue(PriorityQueueBase):
    _data: PositionalList = field(init=False, default=PositionalList())

    def __len__(self) -> int:
        return len(self._data)

    # pyre-fixme
    def add(self, key: Any, value: Any) -> None:
        self._data.add_last(Item(key, value))

    def _find_min(self) -> Optional[Position]:
        if self.is_empty():
            raise Empty("Priority Queue is Empty")
        small = self._data.first()
        walk = self._data.after(small) if small else None
        while walk is not None and small:
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
