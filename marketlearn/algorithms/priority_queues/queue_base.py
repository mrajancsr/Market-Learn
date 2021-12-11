# pyre-strict
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Item:
    key: Any  # pyre-ignore
    value: Any  # pyre-ignore
    index: int

    def __le__(self, other: Item) -> bool:
        return self.value <= other.value

    def __gt__(self, other: Item) -> bool:
        return self.value > other.value


@dataclass
class PriorityQueueBase:
    """Abstract base class for a priority Queue"""

    @abstractmethod
    def __len__(self) -> int:
        pass

    def is_empty(self) -> bool:
        return len(self) == 0
