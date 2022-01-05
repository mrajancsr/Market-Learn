"""Implements the Perceptron & Adaline Learning Algorithm
Author: Rajan Subramanian
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marketlearn.learning.linear_models.base import NeuralBase
from numpy.typing import NDArray


class Perceptron(NeuralBase):
    """Implements the Perceptron Learning Algorithm"""

    def __init__(self, eta: float = 0.01, niter: int = 50, bias: bool = True):
        self.eta = eta
        self.niter = niter
        self.bias = bias
        self.thetas = None
        self.weights = None
        self.degree = 1

    def _fit_batch(self, X: NDArray, y: NDArray) -> Perceptron:
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
        # Add bias unit to design matrix
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree, bias)

        # Initialize weights to 0
        self.thetas = np.zeros(X.shape[1])
        weights = {}
        index = -1
        converged = False

        while not converged:
            index += 1
            prev_weights = self.thetas.copy()
            # for each example in training set
            for xi, target in zip(X, y):
                # update weights if there are misclassifications
                if target != self.predict(xi):
                    self.thetas += target * xi

            weights[index] = self.thetas.copy()
            if (prev_weights == self.thetas).all():
                converged = True
        self.weights = pd.DataFrame.from_dict(
            weights, orient="index", columns=["bias", "weight1", "weight2"]
        )
        return self

    def _fit_online(self, X: NDArray, y: NDArray) -> Perceptron:
        R = (np.sum(np.abs(X) ** 2, axis=-1) ** (0.5)).max()
        bias = 0
        # Initialize weights to 0
        self.thetas = np.zeros(X.shape[1] + 1)
        mistakes = 0
        mistakes_from_previous_iteration = 0
        while True:
            for xi, target in zip(X, y):
                if target * (self.net_input(xi, self.thetas[1:]) + bias) <= 0:
                    self.thetas[1:] += self.eta * target * xi
                    bias += self.eta * target * R * R
                    mistakes += 1
            if mistakes_from_previous_iteration == mistakes:
                break
            mistakes_from_previous_iteration = mistakes

        self.thetas[0] = bias
        return self

    def fit(self, X: NDArray, y: NDArray, learner="batch") -> Perceptron:
        msg = "Incorrect learner type, supports one of batch or online"
        if learner == "batch":
            return self._fit_batch(X, y)
        elif learner == "online":
            return self._fit_online(X, y)
        else:
            raise LookupError(msg)

    def predict(self, X: NDArray, thetas: Optional[NDArray] = None) -> int:
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
        if thetas is None:
            return 1 if self.net_input(X, self.thetas) >= 0 else -1
        return 1 if self.net_input(X, thetas) >= 0 else -1

    def plot_decision_boundary(self, inputs, targets, weights):
        for input, target in zip(inputs, targets):
            plt.plot(input[0], input[1], "ro" if (target == 1.0) else "bo")

        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]
        for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
            y = (slope * i) + intercept
            plt.plot(i, y, "ko")


class Adaline(NeuralBase):
    """Implements the Adaptive Linear Neuron by Bernard Widrow
    via Batch Gradient Descent

    Notes:
        - The cost function is given by J(w) = 1/2 ||(yi - yhat)||
    """

    def __init__(self, eta: float = 0.01, niter: int = 50, bias: bool = True):
        """Default Constructor used to initialize the Adaline model"""
        self.eta = eta
        self.niter = niter
        self.cost = []
        self.bias = bias
        self.thetas = None
        self.degree = 1

    def fit(self, X: NDArray, y: NDArray) -> Adaline:
        """fits training data via batch gradient descent

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
        # add bias + weights for each neuron
        self.thetas = np.zeros(shape=1 + X.shape[1])

        # Add bias unit to design matrix
        degree = self.degree
        X = self.make_polynomial(X, degree=degree, bias=True)

        for _ in range(self.niter):
            error = y - self.activation(self.net_input(X, self.thetas))
            self.thetas += self.eta * np.transpose(X).dot(error)
            self.cost.append((error.transpose() @ error) / 2.0)
        return self

    def activation(self, X: NDArray) -> NDArray:
        """Computes the linear activation function
        given by f(w'x) = w'x

        Parameters
        ----------
        X : np.ndarray
            output from the netinput function

        Returns
        -------
        np.ndarray
            activation function
        """
        return X

    def predict(self, X: NDArray, thetas: Optional[NDArray]) -> float:
        """Computes the class label after activation

        Parameters
        ----------
        X : np.ndarray
            [description]
        thetas : Union[np.ndarray, None], optional
            [description], by default None
        """
        if thetas is None:
            if self.activation(self.net_input(X, self.thetas)) >= 0.0:
                return 1
            else:
                return -1
        else:
            return 1 if self.activation(self.net_input(X, thetas)) >= 0.0 else -1
