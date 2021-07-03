"""Implements the Perceptron & Adaline Learning Algorithm
Author: Rajan Subramanian
"""

from __future__ import annotations
from marketlearn.learning.linear_models.base import NeuralBase
from marketlearn.toolz import timethis
from typing import Union
import matplotlib.pyplot as plt
import numpy as np


class Perceptron(NeuralBase):
    """Implements the Perceptron Learning Algorithm"""

    def __init__(self, eta: float = 0.01, niter: int = 50, bias: bool = True):
        self.eta = eta
        self.niter = niter
        self.errors = None
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
        # Add bias unit to design matrix
        X = self.make_polynomial(X)

        # Generate small random weights
        self.thetas = np.random.rand(X.shape[1])
        self.errors = np.zeros(self.niter)

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
            self.errors[index] = count
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

    def plot_misclassifications(self) -> None:
        """Plots the misclassifications given number of iterations
        Requires call to fit() first, otherwise raise appropriate error

        Raises
        ------
        AttributeError
            if fit() has not been called
        """
        if self.errors is None:
            raise AttributeError(
                "Must call fit() first before plotting \
                    misclassifications"
            )
        # plot the errors
        plt.plot(range(1, self.niter + 1), self.errors, marker="o")
        plt.xlabel("Iterations")
        plt.ylabel("# of misclassifications")
        plt.grid()


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

    @timethis
    def fit(self, X: np.ndarray, y: np.ndarray) -> Adaline:
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
        X = self.make_polynomial(X)

        for _ in range(self.niter):
            error = y - self.activation(self.net_input(X, self.thetas))
            self.thetas += self.eta * X.T @ error
            self.cost.append((error.T @ error) / 2.0)
        return self

    def activation(self, X: np.ndarray) -> np.ndarray:
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

    def predict(
        self, X: np.ndarray, thetas: Union[np.ndarray, None] = None
    ) -> float:
        """Computes the class label after activation

        Parameters
        ----------
        X : np.ndarray
            [description]
        thetas : Union[np.ndarray, None], optional
            [description], by default None
        """
        if thetas is None and self.thetas is None:
            raise ValueError(
                "Empty weights provided, either call fit() first or provide \
                    weights"
            )
        elif thetas is None:
            return (
                1
                if self.activation(self.net_input(X, self.thetas)) >= 0.0
                else -1
            )
        return 1 if self.activation(self.net_input(X, thetas)) >= 0.0 else -1
