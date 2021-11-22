from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from base import NeuralBase

INPUT_PATH_TO_FILE = os.path.join(
    os.getcwd(), "artificial_intelligence", "hw4", "starter", "data1.csv"
)

OUTPUT_PATH_TO_FILE = os.path.join(
    os.getcwd(), "artificial_intelligence", "hw4", "starter", "results1.csv"
)


@dataclass
class Perceptron(NeuralBase):
    """Implements the Perceptron Learning Algorithm"""

    def __init__(self, eta: float = 0.01, niter: int = 50, bias: bool = True):
        self.eta = eta
        self.niter = niter
        self.bias = bias
        self.errors = None
        self.thetas = None
        self.degree = 1

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
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree, bias)

        # Generate small random weights
        self.thetas = np.random.rand(X.shape[1])
        self.errors = np.zeros(self.niter)
        weights = {}

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
            # updated weight per iteration
            weights[index] = self.thetas.copy()
            # store count of errors in each iteration
            self.errors[index] = count

        self.weights = pd.DataFrame.from_dict(
            weights, orient="index", columns=["bias", "coef1", "coef2"]
        )

        return self

    def predict(
        self, X: np.ndarray, thetas: Optional[np.ndarray] = None
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
        plt.show()


def plot_data(inputs, targets, weights):
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    for input, target in zip(inputs, targets):
        plt.plot(input[0], input[1], "ro" if (target == 1.0) else "bo")

    for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
        slope = -weights[1] / weights[2]
        intercept = -weights[0] / weights[2]
        y = (slope * i) + intercept
        plt.plot(i, y, "ko")
    plt.show()


def main():
    """YOUR CODE GOES HERE"""
    pla = Perceptron(eta=0.1)
    data = np.genfromtxt(INPUT_PATH_TO_FILE, delimiter=",")
    inputs, targets = data[:, :2], data[:, 2]
    pla.fit(inputs, targets)
    weight = pla.thetas
    plot_data(inputs, targets, weight)

    pla.weights.to_csv(OUTPUT_PATH_TO_FILE)


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()
