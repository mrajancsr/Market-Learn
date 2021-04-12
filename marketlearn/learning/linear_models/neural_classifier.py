"""Implements the Perceptron & Adaline Learning Algorithm
Author: Rajan Subramanian
"""

from __future__ import annotations
from marketlearn.learning.linear_models.base import NeuralBase
from marketlearn.toolz import timethis
import numpy as np


class Perceptron(NeuralBase):
    """Implements the Perception Learning Algorithm"""

    def __init__(self, eta: float = 0.01, niter: int = 50):
        self.eta = eta
        self.niter = niter
        self.cost = None

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
        # generate random numbers
        thetas = np.random.rand(X.shape[1])
        self.cost = np.zeros(self.niter)

        for _ in range(self.niter):
            # for each instance in the training set
            for xi, target in zip(X, y):
                pass


percept_obj = Perceptron()
