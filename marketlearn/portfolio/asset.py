# pyre-strict
"""Implementation of Asset class"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import ClassVar, List, Tuple

import numpy as np
import pandas as pd
from numpy import array, cov, isnan, log
from numpy.typing import NDArray

__TRADING_DAYS_PER_YEAR: int = 252


@dataclass
class Asset:
    name: str
    price_history: pd.Series
    size: int = field(init=False)
    returns_history: pd.Series = field(init=False)
    annualized_returns: float = field(init=False)
    expected_returns: float = field(init=False)
    all_assets: ClassVar[List[Asset]] = []

    def __post_init__(self) -> None:
        self.size = self.price_history.shape[0]
        self.returns_history = log(1 + self.price_history.pct_change())
        self.annualized_returns = self.returns_history.sum()
        self.expected_returns = self._get_expected_returns()
        self.__class__.all_assets.append(self)

    def __hash__(self) -> int:
        """allows hashing for lru_cache

        Returns
        -------
        hashid
            hashid based on asset name
        """
        return hash(self.name)

    def __eq__(self, other: Asset) -> bool:
        return type(self) is type(other) and self.name == other.name

    def __ne__(self, other: Asset) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        return f"Asset name: {self.name}, \
        \nexpected returns: {self.expected_returns:.5f}, \
        \nannualized_returns: {self.annualized_returns:.5f}"

    def _get_expected_returns(self) -> float:
        return Asset.get_annualization_factor() * self.returns_history.mean()

    @staticmethod
    def get_annualization_factor() -> int:
        return __TRADING_DAYS_PER_YEAR

    @staticmethod
    @lru_cache
    def covariance_matrix(
        assets: Tuple[Asset],
    ) -> NDArray[np.float64]:
        """computes the covariance matrix given tuple of assets

        Parameters
        ----------
        assets : Tuple[Asset]
            Assets whose covariance we want to compute

        Returns
        -------
        np.ndarray
            covariance matrix of assets
        """
        returns = array([asset.returns_history for asset in assets]).T
        returns = returns[~isnan(returns).any(axis=1)]
        return cov(returns, rowvar=False) * Asset.get_annualization_factor()
