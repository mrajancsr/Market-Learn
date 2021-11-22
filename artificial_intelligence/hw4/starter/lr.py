from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from base import LinearBase

INPUT_PATH_TO_FILE = os.path.join(
    os.getcwd(), "artificial_intelligence", "hw4", "starter", "data2.csv"
)

OUTPUT_PATH_TO_FILE = os.path.join(
    os.getcwd(), "artificial_intelligence", "hw4", "starter", "results2.csv"
)


@dataclass()
class LinearRegressionGD(LinearBase):
    """Implements the ols regression via Gradient Descent

    Args:
    eta:             Learning rate (between 0.0 and 1.0)
    n_iter:          passees over the training set
    random_state:    Random Number Generator seed
                     for random weight initilization

    Attributes:
    theta:           Weights after fitting
    residuals:       Number of incorrect predictions
    """

    eta: float = 0.001
    n_iter: int = 20
    random_state: int = 1
    bias: bool = True
    degree: int = 1
    cost: List[float] = field(init=False)
    theta: Optional[np.ndarray] = field(init=False)
    run: bool = field(init=False, default=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionGD:
        """Fits model to training data via Gradient Descent

        Parameters
        ----------
        X : ArrayLike
            [description]
        y : ArrayLike
            [description]

        Returns
        -------
        LinearRegressionGD
            [description]
        """
        n_samples, p_features = X.shape[0], X.shape[1]
        self.theta = np.zeros(shape=1 + p_features)
        self.cost = []
        degree, bias = self.degree, self.bias
        X = self.make_polynomial(X, degree=degree, bias=bias)

        for _ in range(self.n_iter):
            # calculate the error
            error = y - self.predict(X)
            self.theta += self.eta * X.T @ error / n_samples
            self.cost.append((error.T @ error) / (2.0 * n_samples))
        self.run = True
        return self

    def predict(
        self, X: np.ndarray, thetas: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Makes predictions of target variable given data

        Parameters
        ----------
        X : ArrayLike, shape=(n_samples, p_features)
            [description]
        thetas : Optional[ArrayLike], optional, default=None
            weights of parameters in model

        Returns
        -------
        NDArray
            predictions of response given thetas
        """
        if thetas is None:
            return X @ self.theta
        return X @ thetas


def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    data = np.genfromtxt(INPUT_PATH_TO_FILE, delimiter=",")
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    target = data[:, -1]
    inputs = data[:, :2]

    lg = LinearRegressionGD(n_iter=100)

    learning_rates = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10)
    weights = {}
    for idx, eta in enumerate(learning_rates):
        lg.eta = eta
        _ = lg.fit(inputs, target)
        weights[idx] = (idx + 1, 100, eta, *lg.theta)
    weights = pd.DataFrame.from_dict(weights, orient="index")
    weights.to_csv(OUTPUT_PATH_TO_FILE)


if __name__ == "__main__":
    main()
