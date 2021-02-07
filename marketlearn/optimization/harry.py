"""
Implementation of Harry Markowitz's Modern Portfolio Theory
Author: Rajan Subramanian
Created Feb 10 2021
"""

import numpy as np


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

    def __init__(self, means, sigmas, cov):
        self.means = means
        self.sigmas = sigmas
        self.cov = cov
        self.weights = np.zeros_like(means)
        self.size = len(means)

    def get_portfolio_mean(self) -> float:
        """Computes the portfolio mean

        Returns
        -------
        float
            portfolio mean
        """
        return self.weights.T @ self.means

    def get_portfolio_variance(self) -> float:
        """Computes the portfolio variance

        Returns
        -------
        float
            [description]
        """
        return self.weights.T @ self.cov @ self.weights

    def global_minimum_variance(self, mean_constraint=False, target=None) -> np.ndarray:
        """Computes the weights corresponding to global minimum variance

        Parameters
        ----------
        mean_constraint : bool, optional
            if true, add the constraint w'mu=target
        target : float, optional, default=None
            if mean_constraint is True, add target

        Returns
        -------
        np.ndarray
            weights corresponding to minimum variance
        """
