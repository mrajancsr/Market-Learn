"""
Abstract base class for Vector AutoRegressive Models
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# pylint: disable=invalid-name
class Base(metaclass=ABCMeta):
    """
    Abstract Base class representing the VAR model
    """

    @abstractmethod
    def fit(self, X, y, method):
        """
        Future fit function.
        """

    @abstractmethod
    def predict(self, X, thetas):
        """
        Future predict function
        """

    def make_polynomial(self, X: np.ndarray,
                        fit_intercept: bool = None) -> np.ndarray:
        """
        Fits an intercept and creates a polynomial of degree d


        :param X: (np.array) The design matrix with shape (n_samples, p_features)
        :param fit_intercept: (bool) Flag to add bias (None by default)
        :return: (np.array) Polynomial
        """
        # Use inherited objects intercept if not supplied
        if fit_intercept:
            bias = fit_intercept
        else:
            bias = self.fit_intercept

        degree = self.degree
        pf = PolynomialFeatures(degree=degree, include_bias=bias)

        return pf.fit_transform(X)
