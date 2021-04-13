"""
Implementation of Harry Markowitz's Modern Portfolio Theory
Author: Rajan Subramanian
Created Feb 10 2021
"""

from __future__ import annotations
from marketlearn.optimization import Asset
import numpy as np
import pandas as pd


class Harry:
    """
    Implements Harry Markowitz'a Model of mean variance optimization

    Parameters
    ----------
    means : np.ndarray, shape=(p_constituents,)
        represents forecasted means of constituents in portfolio
    sigmas : np.ndarray, shape=(p_constituents,)
        represents forecasted volatility of constituents in portfolio
    cov : np.ndarray, shape=(p_constituents, p_constituents)
        covariance matrix of constituents in portfolio

    Attributes
    ----------
    weights: np.ndarray
        proportion of capital allocated to security i
    size: np.ndarray
        total number of securities in portfolio
    """

    def __init__(self, historical_prices: pd.DataFrame):
        self.assets = tuple(
            Asset(name=name, price_history=historical_prices[name])
            for name in historical_prices.columns
        )
        self.covariance_matrix = Asset.covariance_matrix(self.assets)
        self.asset_expected_returns = np.fromiter(
            self.get_asset_expected_returns(), dtype=float
        )
        self.asset_expected_vol = np.fromiter(
            self.get_asset_expected_volatility(), dtype=float
        )

    def get_assets(self):
        yield from self.assets

    def get_asset_expected_returns(self):
        for asset in self.get_assets():
            yield asset.expected_returns

    def get_asset_expected_volatility(self):
        vols = np.sqrt(
            np.diag(self.covariance_matrix) * Asset.get_annualization_factor()
        )
        yield from vols

    def portfolio_mean(self):
        pass