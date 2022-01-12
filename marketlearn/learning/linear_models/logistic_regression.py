# pyre-strict
"""Linear Classification Models
Author: Rajan Subramanian
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union

import numpy as np
from marketlearn.learning.linear_models.base import LogisticBase
from marketlearn.toolz import timethis
from numpy import fill_diagonal, float64
from numpy.linalg import solve
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


@dataclass
class LogisticRegression(LogisticBase):
    """
    Implements Logistic Regression via MLE
    Args:
    fit_intercept: indicates if intercept is needed or not

    Attributes:
    theta:          Coefficient Weights after fitting
    predictions:    Predicted Values from fitting
    residuals:      Number of incorrect Predictions

    Notes:
    Class uses multiple methods to estimate the parameters
    of logistic regression
    - A implemention using BFGS using scipy's optimizer to solve the MLE
    - A implementation using Iterative Reweighted Least Squares (IRLS) to
      estimate the parameters of MLE
      see rf. Bishop - "Machine Learning - A probabilistic perspective"
      For IRLS algorithm, I have chosen to store the
      diagonal matrix of probabilities and its inverse in the
      same matrix.  This avoids an extra O(n^2) storage for storing
      the inverse at slight increase in expense of refilling the original
      matrix in O(n) time through the iterative method
      Also, i have chosen to use numpy's broadcasting feature vs
      computing the calculations per sample.
    - todo: A implemention using Stochastic Gradient Descent
    """

    fit_intercept: bool = True
    degree: int = 1
    run: bool = field(init=False)

    def __post_init__(self) -> None:
        self.run = False

    def _jacobian(
        self, guess: NDArray[float64], X: NDArray[float64], y: NDArray[float64]
    ) -> ArrayLike:
        """Computes the Jacobian of likelihood function

        Parameters
        ----------
        guess : np.ndarray
            initial guess for optimizer
        X : np.ndarray, shape=(n_samples, p_features)
            design matrix
        y : np.ndarray, shape=(n_samples)
            response variable

        Returns
        -------
        np.ndarray, shape=(intercept + p_features,)
            first partial derivatives wrt weights
        """
        predictions = self.predict(X, guess)
        return -1 * X.T @ (y - predictions)

    def _hessian(self, X: NDArray, W: NDArray) -> NDArray:
        """Computes the Hessian wrt to weights

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, p_features)
            design matrix
        W : np.ndarray, shape=(n_samples, n_samples)
            diagonal matrix whose elements
            are given by P(Yi=1) * (1 - P(Yi = 1))

        Returns
        -------
        np.ndarray, shape=(p_features, p_features)
            Hessian matrix
        """

        return X.T @ W @ X

    def _loglikelihood(self, y: NDArray, z: NDArray) -> float:
        """Returns loglikelihood function of logistic regression

        Parameters
        ----------
        y : np.ndarray
            response variable
        z : np.ndarray
            result of net input function

        Returns
        -------
        float
            loglikelihood function
        """
        return y @ z - np.log(1 + np.exp(z)).sum()

    def _irls(self, X: NDArray, y: NDArray, niter: int = 20) -> NDArray:
        """performs iteratively reweighted least squares

        Parameters
        ----------
        X : np.ndarray
            design matrix
        y : np.ndarray
            response

        Returns
        -------
        np.ndarray
            weights from performing the reweighted algorithm
        """
        n = X.shape[0]
        guess = np.zeros(X.shape[1])
        # will be used to store the predictions and its inverse
        W = np.zeros((n, n))
        Winv = np.zeros((n, n))
        # --- currently using O(n^2) storage
        for _ in range(niter):
            prob = self.predict(X, guess)
            fill_diagonal(W, prob * (1 - prob))
            H = self._hessian(X, W)
            fill_diagonal(Winv, 1 / (prob * (1 - prob)))
            zbar = X @ guess + Winv @ (y - prob)
            # fill_diagonal(W, prob * (1 - prob))
            guess = solve(H, X.T @ W @ zbar)
        return guess

    def _sgd(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        """performs stochastic gradient descent to estimate weights

        Parameters
        ----------
        guess : np.ndarray
            [description]
        X : np.ndarray
            [description]
        y : np.ndarray
            [description]
        """
        pass

    def _objective_func(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        """the objective function to be minimized

        Parameters
        ----------
        guess : np.ndarray
            initial guess for optimizer
        X : np.ndarray
            design matrix
        y : np.ndarray
            response variable

        Returns
        -------
        float
            minimum of negative likelihood function
        """
        z = self.net_input(X, thetas=guess)
        f = self._loglikelihood(y, z)
        return -f

    @timethis
    def fit(
        self, X: np.ndarray, y: np.ndarray, fit_type="BFGS", niter=20
    ) -> LogisticRegression:
        """fits model to training data and returns regression coefficients

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, p_features)
            Design matrix
            n_samples is number of instances i.e rows
            p_features is number of features
        y : np.ndarray, shape=(n_samples)
            Target values
        fit_type : str, optional, default='BFGS'
            supports 'BFGS' and 'IRLS' algorithm
        niter : int, optional, default=20
            number of iterations to perform if 'IRLS' is chosen

        Returns
        -------
        LogisticRegression
            fitted object
        """
        X = self.make_polynomial(X)
        if fit_type == "IRLS":
            self.theta = self._irls(X, y, niter=niter)
            return self

        guess_params = np.zeros(X.shape[1])
        self.theta = minimize(
            self._objective_func,
            guess_params,
            jac=self._jacobian,
            method=fit_type,
            options={"disp": True},
            args=(X, y),
        )["x"]
        return self

    def predict(
        self,
        X: np.ndarray,
        thetas: np.ndarray = None,
    ) -> Union[np.ndarray, Dict]:
        """Makes predictions of probabilities

        Args:
            X (np.ndarray): design matrix
            shape = {n_samples, p_features}
            thetas (np.ndarray, optional): estimated weights from fitting
            Defaults to None.
            shape = {p_features + intercept,}

        Returns:
            Union[np.ndarray, Dict]: predicted probabilities
            shape = {n_samples,}
        """
        if thetas is None:
            if isinstance(self.theta, np.ndarray):
                return self.sigmoid(self.net_input(X, self.theta))
            else:
                return self.sigmoid(self.net_input(X, self.theta["x"]))
        return self.sigmoid(self.net_input(X, thetas))
