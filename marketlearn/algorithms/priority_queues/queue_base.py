# pyre-strict
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Item:
    # pyre-fixme
    key: Any
    # pyre-fixme
    value: Any

    def __lt__(self, other: Item) -> bool:
        return self.key <= other.key


@dataclass
class PriorityQueueBase:
    """Abstract base class for a priority Queue"""

    @abstractmethod
    def __len__(self) -> int:
        pass

    def is_empty(self) -> bool:
        return len(self) == 0
