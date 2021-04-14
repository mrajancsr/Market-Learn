"""
Implementation of Harry Markowitz's Modern Portfolio Theory
Author: Rajan Subramanian
Created Feb 10 2021
"""

from __future__ import annotations
from marketlearn.optimization import Asset
from numpy import fromiter
from numpy.random import random
from typing import Iterator
import numpy as np
import pandas as pd


class Harry:
    """
    Implements Harry Markowitz'a Model of mean variance optimization

    Parameters
    ----------
    historical_prices : pd.DataFrame
        represents historical daily end of day prices

    Attributes
    ----------
    assets: Dict[name: Asset]
        dictionary of Asset objects whose keys are asset names
    covariance_matrix: np.ndarray
        Scaled Covariance Matrix of daily log returns
    asset_expected_returns: np.ndarray
        forecasted sample means of each Asset
    asset_expected_vol: np.ndarray
        forecasted sample volatility of each Asset
    """

    def __init__(self, historical_prices: pd.DataFrame):
        """Default constructor used to initialize portfolio"""
        self._assets = {
            name: Asset(name=name, price_history=historical_prices)
            for name in historical_prices.columns
        }
        self.covariance_matrix = Asset.covariance_matrix(self._assets.values())
        self.asset_expected_returns = fromiter(
            self.get_asset_expected_returns(), dtype=float
        )
        self.asset_expected_vol = fromiter(
            self.get_asset_expected_volatility(), dtype=float
        )

    def __iter__(self):
        yield from self._assets.values()

    def get_asset(self, name: str) -> Asset:
        """return the Asset in portfolio given name

        Parameters
        ----------
        name : str
            name of the Asset

        Returns
        -------
        Asset
            contains information about the asset
        """
        return self._assets[name]

    def get_assets(self) -> Iterator[Asset]:
        """return iteration of assets in a portfolio

        Yields
        -------
        Iterator[Asset]
           returns all assets in the portfolio
        """
        yield from self

    def get_asset_expected_returns(self):
        """gets expected return of each asset in portfolio

        Yields
        -------
        Iterator[float]
            an iteration of expected return of each asset in portfolio
        """
        yield from (asset.expected_returns for asset in self)

    def get_asset_expected_volatility(self):
        """gets expected volatility of each asset in portfolio

        Yields
        -------
        Iterator[float]
            iteration of expected volatility of each asset scaled by trading days
        """
        yield from np.sqrt(
            np.diag(self.covariance_matrix) * Asset.get_annualization_factor()
        )

    @staticmethod
    def random_weights(nsec: int, nsim: int = 1):
        """creates a portfolio with random weights

        Parameters
        ----------
        nsec : int
            number of securities in porfolio
        nsim : int, optional, default=1
            number of simulations to perform

        Returns
        -------
        np.ndarray, shape=(nsim, nsec)
            random weight matrix
        """
        weights = random((nsim, nsec)) if nsim != 1 else random(nsec)
        return weights / weights.sum(axis=0)
