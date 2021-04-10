"""Linear Classification Models
Author: Rajan Subramanian
Created: May 25, 2020
"""

from __future__ import annotations
from scipy.optimize import minimize
from marketlearn.learning.linear_models.base import LogisticBase
from typing import Union, Dict, Callable
from numpy.linalg import norm, solve
from numpy import diagonal, diagflat, copyto, fill_diagonal
import numpy as np
from marketlearn.toolz import timethis


class LogisticRegressionMLE(LogisticBase):
    """
    Implements Logistic Regression via MLE
    Args:
    fit_intercept: indicates if intercept is needed or not

    Attributes:
    theta:          Coefficient Weights after fitting
    predictions:    Predicted Values from fitting
    residuals:      Number of incorrect Predictions

    Notes:
    Class uses two estimation methods to estimate the parameters
    of logistic regression
    - A implemention using Newton's Method is given
    - A implemention using Stochastic Gradient Descent is given
    """

    def __init__(self, fit_intercept: bool = True, degree: int = 1):
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.run = False

    def _jacobian(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Computes the jacobian of likelihood function

        Args:
            guess (np.ndarray): the initial guess for optimizer
            X (np.ndarray): design matrix
                            shape = (n_samples, n_features)
            y (np.ndarray): response variable
                            shape = (n_samples,)

        Returns:
            [np.ndarray]: first partial derivatives wrt weights
        """
        predictions = self.predict(X, guess)
        return -1 * X.T @ (y - predictions)

    def _hessian(
        self, guess: np.ndarray, X: np.ndarray, y: np.ndarray, full_list=False
    ):
        """computes the hessian wrt weights

        Args:
            guess (np.ndarray): initial guess for optimizer
            X (np.ndarray): design matrix
                            shape = (n_samples, n_features)
            y (np.ndarray): response variable
                            shape = (n_samples,)

        Returns:
            np.ndarray: second partial derivatives wrt weights
        """
        prob = self.predict(X, guess)
        W = np.diagflat(prob * (1 - prob))
        return X.T @ W @ X, W, prob if full_list else X.T @ W @ X

    def _loglikelihood(self, y: np.ndarray, z: np.ndarray):
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

    @timethis
    def _irls(self, X: np.ndarray, y: np.ndarray, niter=20) -> np.ndarray:
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
        W = np.zeros((n, n))
        Winv = np.zeros((n, n))

        for _ in range(niter):
            H, W, prob = self._hessian(guess, X, y, full_list=True)
            fill_diagonal(Winv, 1 / (prob * (1 - prob)))
            zbar = X @ guess + Winv @ (y - prob)
            guess = np.linalg.inv(H).dot(X.T).dot(W).dot(zbar)
        return guess

    def _objective_func(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        """the objective function to be minimized

        Args:
            guess (np.ndarray): initial guess for optimization
            X (np.ndarray): design matrix
                            shape = {n_samples, p_features}
            y (np.ndarray): the response variable
                            shape = {n_samples,}

        Returns:
            float: value from loglikelihood function
        """
        z = self.net_input(X, thetas=guess)
        f = self._loglikelihood(y, z)
        return -f

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> LogisticRegressionMLE:
        """fits model to training data and returns regression coefficients
        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y:
            shape = (n_samples,)
            Target values

        Returns:
        object
        """
        X = self.make_polynomial(X)
        # generate random guess
        guess_params = np.zeros(X.shape[1])
        self.theta = minimize(
            self._objective_func,
            guess_params,
            jac=self._jacobian,
            method="BFGS",
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