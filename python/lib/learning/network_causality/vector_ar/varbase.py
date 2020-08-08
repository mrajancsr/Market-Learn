"""Abstract base class for Vector AutoRegressive Models"""

from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class VarBase(metaclass=ABCMeta):
    """Abstract Base class representing the VAR model"""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def simulate_var(self, n_samples=1000, corr=0.8):
        """Simulates bivariate vector autoregressive model of order 1

        Args:
            n_samples (int): Number of observations to use
            Defaults to 1000.
            corr (float): correlation between two variables.
            Defaults to 0.8

        Returns:
            [tuple]: two vectors with given correlation
        """
        cov_matrix = np.array([[1, corr], [corr, 1]])
        mean_vector = np.zeros(2)
        w = np.random.multivariate_normal(
                                         mean=mean_vector,
                                         cov=cov_matrix,
                                         size=n_samples)
        x = np.zeros(n_samples)
        y = np.zeros(n_samples)
        wx = w[:, 0]
        wy = w[:, 1]
        x[0] = wx[0]
        y[0] = wy[0]
        for i in range(1, n_samples):
            x[i] = 0.1 + 0.01 * x[i-1] + 0.3 * y[i-1] + wx[i]
            y[i] = 0.1 + 0.3 * x[i-1] + 0.01 * y[i-1] + wy[i]
        return (x, y)

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
