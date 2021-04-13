"""Implementation of Asset class"""

from __future__ import annotations
from functools import lru_cache
from numpy import array
from marketlearn.toolz import timethis
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf


class Asset:
    """Serves as a composite class to Portfolio Class"""

    all_assets = []

    __TRADING_DAYS_PER_YEAR = 252

    def __init__(self, name: str, price_history: pd.Series):
        """default constructor used to initialize Asset Class"""
        self.name = name
        self.price_history = price_history
        self.size = price_history.shape[0]
        self.returns_history = np.log(1 + self.price_history.pct_change())
        self.annualized_returns = self.returns_history.sum()
        self.expected_returns = self._get_expected_returns()
        self.__class__.all_assets.append(self)

    def _get_expected_returns(self):
        return Asset.__TRADING_DAYS_PER_YEAR * self.returns_history.mean()

    @staticmethod
    def get_annualization_factor():
        return Asset.__TRADING_DAYS_PER_YEAR

    def __repr__(self):
        return f"Asset name: {self.name}, expected returns: {self.expected_returns}"

    @classmethod
    @lru_cache
    def covariance_matrix(cls, assets: Tuple[Asset]):
        zt = array([c.returns_history - c.expected_returns for c in cls.all_assets])
        return np.cov(zt.T[~np.isnan(zt.T).any(axis=1)], rowvar=False)
