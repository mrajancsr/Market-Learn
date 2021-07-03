"""Abstract base class for regression models"""

from abc import ABCMeta, abstractmethod
from typing import Dict
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt


class LinearBase(metaclass=ABCMeta):
    """Abstract Base class representing the Linear Model"""

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def make_regression_example(
        self, n_samples: int = 1000, n_features: int = 5
    ) -> Dict:
        features, output, coef = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_targets=1,
            noise=5,
            coef=True,
        )
        return dict(zip(["X", "y", "coef"], [features, output, coef]))

    def make_polynomial(self, X: np.ndarray) -> np.ndarray:
        degree, bias = self.degree, self.bias
        pf = PolynomialFeatures(degree=degree, include_bias=bias)
        return pf.fit_transform(X)

    def reg_plot(self, X: np.ndarray, y: np.ndarray):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y)
        # sort by design matrix -- needed for matplotlib
        sorted_values = iter(
            sorted(zip(X.flatten(), self.predictions), key=itemgetter(0))
        )
        X, pred = zip(*sorted_values)
        plt.plot(X, pred, "m-")
        plt.title("Regression Plot")


class LogisticBase(LinearBase):
    """Abstract Base class representing a Logistic Regression Model"""

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function

        Parameters
        ----------
        z : np.ndarray
            input value from linear transformation

        Returns
        -------
        np.ndarray
            sigmoid function value
        """
        return 1.0 / (1 + np.exp(-z))

    def net_input(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Computes the Linear transformation X@theta

        Parameters
        ----------
        X : np.ndarray, shape={n_samples, p_features}
            design matrix
        thetas : np.ndarray, shape={p_features + intercept}
            weights of logistic regression

        Returns
        -------
        np.ndarray
            linear transformation
        """
        return X @ thetas


class NeuralBase(LinearBase):
    """Abstract Base class representing a Neural Network"""

    def net_input(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Computes the net input vector
        z = w1x1 + w2x2 + ... + wpxp := w'x

        Parameters
        ----------
        X : np.ndarray, shape={n_samples, p_features}
            design matrix
        thetas : np.ndarray, shape={p_features + intercept}
            weights of neural classifier, w vector above
            assumes first element is the bias unit i.e intercept

        Returns
        -------
        np.ndarray
            linear transformation
        """
        return X @ thetas
