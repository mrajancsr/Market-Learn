"""
Implementation of Modern Portfolio Theory
Author: Rajan Subramanian 
Created May 15 2020
"""

import numpy as np


class Harry:
    """
    Implements Harry Markowitz'a Model of mean variance optimization

    Parameters
    ----------
    mu_vec : np.ndarray
        represents forecasted means of constituents in portfolio
    sigma_vec : np.ndarray
        represents forecasted volatility of constituents in portfolio
    cov_mat : np.ndarray
        covariance matrix of constituents in portfolio

    Attributes
    ----------
    weights: np.ndarray
        proportion of capital allocated to security i
    size: np.ndarray
        total number of securities in portfolio
    """

    def __init__(self, mu_vec: np.ndarray, sigma_vec: np.ndarray, cov_mat: np.ndarray):
        self.mu_vec = mu_vec
        self.sigma_vec = sigma_vec
        self.cov_mat = cov_mat
        self.size = len(mu_vec)

    def compute_portfolio_mean(self) -> float:
        pass
