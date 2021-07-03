"""Implements the Perceptron & Adaline Learning Algorithm
Author: Rajan Subramanian
"""

from __future__ import annotations
from marketlearn.learning.linear_models.base import NeuralBase
from marketlearn.toolz import timethis
from typing import Union
import numpy as np


class Perceptron(NeuralBase):
    """Implements the Perceptron Learning Algorithm"""

    def __init__(self, eta: float = 0.01, niter: int = 50, bias: bool = True):
        self.eta = eta
        self.niter = niter
        self.error = None
        self.bias = bias
        self.thetas = None
        self.degree = 1

    @timethis
    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        """fits training data

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features (dimension of data)
        y : np.ndarray
            response variable

        Returns
        -------
        Perception
            object with fitted parameters
        """

        # Generate small random weights
        self.thetas = np.random.rand(X.shape[1])
        self.error = np.zeros(self.niter)

        # Add bias unit to design matrix
        X = self.make_polynomial(X)

        for index in range(self.niter):
            # Count total misclassifications in each iteration
            count = 0

            # Iterate through each example and identify misclassifications
            # Number of errors must decline after each iteration
            for xi, target in zip(X, y):
                # make prediction
                yhat = self.predict(xi)

                # update weights if there are misclassifications
                if target * yhat <= 0:
                    self.thetas += self.eta * (target - yhat) * xi
                    count += 1

            # store count of errors in each iteration
            self.error[index] = count
        return self

    def predict(
        self, X: np.ndarray, thetas: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Activation function to determine if neuron should fire or not

        Parameters
        ----------
        X : np.ndarray
            design matrix that includes the bias
        thetas : Union[np.ndarray, None], optional
            weights from fitting, by default None

        Returns
        -------
        np.ndarray
            predictions
        """
        if thetas is None and self.thetas is None:
            raise ValueError(
                "Empty weights provided, either call fit() first or provide \
                    weights"
            )
        elif thetas is None:
            return 1 if self.net_input(X, self.thetas) >= 0 else -1
        return 1 if self.net_input(X, thetas) >= 0 else -1
