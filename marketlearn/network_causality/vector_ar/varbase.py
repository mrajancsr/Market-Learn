"""Abstract base class for Vector AutoRegressive Models"""

from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Base(metaclass=ABCMeta):
    """Abstract Base class representing the VAR model"""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def make_polynomial(self,
                        X: np.ndarray,
                        fit_intercept: bool = None) -> np.ndarray:
        """Fits an intercept and creates a polynomial of degree d

        Args:
            X (np.ndarray): the design matrix
            shape = (n_samples, p_features)
            fit_intercept (bool, optional): whether to add bias or not.
            Defaults to None.

        Returns:
            np.ndarray: [description]
        """
        # use inherited objects intercept if not supplied
        if fit_intercept is None:
            bias = self.fit_intercept
        else:
            bias = fit_intercept
        degree = self.degree
        pf = PolynomialFeatures(degree=degree, include_bias=bias)
        return pf.fit_transform(X)
