"""Linear Classification Models
Author: Rajan Subramanian
Created: May 25, 2020
"""

from __future__ import annotations
from scipy.optimize import minimize
from marketlearn.learning.linear_models.base import LogisticBase
from typing import Union, Dict
from numpy.linalg import solve
from numpy import fill_diagonal
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
    - A implemention using BFGS using scipy's optimizer to solve the MLE is given
    - A implementation using Iterative Reweighted Least Squares (IRLS) to
      estimate the parameters of MLE is given
      see rf. Bishop - Machine Learning - A probabilistic perspective
      Note that for IRLS algorithm, I have chosen to store the
      diagonal matrix of probabilities and its inverse in the
      same matrix.  This avoids an extra O(n^2) storage for storing
      the inverse at slight increase in expense of refilling the original
      matrix in O(n) time through the iterative method
      Also, i have chosen to use numpy's broadcasting feature vs
      computing the calculations per sample.
    - A implemention using Stochastic Gradient Descent is given
    """

    def __init__(self, fit_intercept: bool = True, degree: int = 1):
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.run = False

    def _jacobian(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
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

    def _hessian(self, X: np.ndarray, W: np.ndarray):
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

        return X.T @ W @ X

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
        # will be used to store the predictions and its inverse
        W = np.zeros((n, n))
        Winv = np.zeros((n, n))

        for _ in range(niter):
            prob = self.predict(X, guess)
            fill_diagonal(W, prob * (1 - prob))
            H = self._hessian(X, W)
            fill_diagonal(Winv, 1 / (prob * (1 - prob)))
            zbar = X @ guess + Winv @ (y - prob)
            # fill_diagonal(W, prob * (1 - prob))
            guess = solve(H, X.T @ W @ zbar)
        return guess

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
