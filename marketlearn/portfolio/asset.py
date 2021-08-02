# pyre-strict
"""Implementation of Asset class"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy import array, cov, isnan, log, transpose


class Asset:
    """Serves as a composite class to Portfolio Class"""

    all_assets: List[Asset] = []

    __TRADING_DAYS_PER_YEAR = 252

    def __init__(self, name: str, price_history: pd.Series) -> None:
        """default constructor used to initialize Asset Class"""
        self.name = name
        self.price_history = price_history
        self.size: int = price_history.shape[0]
        self.returns_history: pd.Series = log(
            1 + self.price_history.pct_change()
        )
        self.annualized_returns: int = self.returns_history.sum()
        self.expected_returns: float = self._get_expected_returns()
        self.__class__.all_assets.append(self)

    def _get_expected_returns(self) -> float:
        return Asset.get_annualization_factor() * self.returns_history.mean()

    @staticmethod
    def get_annualization_factor() -> int:
        return Asset.__TRADING_DAYS_PER_YEAR

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

    @staticmethod
    @lru_cache
    def covariance_matrix(
        assets: Tuple[Asset],
    ) -> npt.NDArray[np.float64]:
        """computes the covariance matrix given tuple of assets

        Parameters
        ----------
        assets : Tuple[Asset]
            Assets whose covariance we want to compute

        Returns
        -------ß
        np.ndarray
            covariance matrix of assets
        """
        returns = transpose(array([asset.returns_history for asset in assets]))
        returns = returns[~isnan(returns).any(axis=1)]
        return cov(returns, rowvar=False) * Asset.get_annualization_factor()
