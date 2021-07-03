"""Implementation of Linear Regression using various fitting methods
Author: Rajan Subramanian
Created: May 23, 2020
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from typing import Dict, Union
from marketlearn.learning.linear_models.base import LinearBase


class LinearRegression(LinearBase):
    """
    Implements the classic Linear Regression via ols
    Args:
    bias: indicates if intercept is added or not

    Attributes:
    theta:          Coefficient Weights after fitting
    residuals:      Number of Incorrect Predictions
    rss:            Residual sum of squares given by e'e
    tss:            Total sum of squares
    ess:            explained sum of squares
    r2:             Rsquared or proportion of variance
    s2:             Residual Standard error or RSE


    Notes:
    Class uses multiple estimation methods to estimate the oridiinary
    lease squares problem min ||Ax - b||, where x = px1 is the parameter
    to be estimated, A=nxp matrix and b = nx1 vector is given
    - A naive implementation of (A'A)^-1 A'b = x is given
      but computing an inverse is expensive
    - A implementation based on QR decomposition is given based on
        min||Ax-b|| = min||Q'(QRx - b)|| = min||(Rx - Q'b)
        based on decomposing nxp matrix A = QR, Q is orthogonal, R is upper
        triangular
    - A cholesky implementation is also included based on converting an n x p
        into a pxp matrix: A'A = A'b, then letting M = A'A & y = A'b, then
        solve Mx = y.  Leting M = U'U, we solve this by forward/backward sub
    """

    def __init__(self, bias: bool = True, degree: int = 1):
        self.bias = bias
        self.degree = degree
        self.run = False

    def _linear_solve(
        self, A: np.ndarray, b: np.ndarray, method: str = "ols-cholesky"
    ) -> np.ndarray:
        """numerically solves Ax = b where x is the parameters to be determined
        based on ||Ax - b||

        Parameters
        ----------
        A : np.ndarray, shape=(n_samples, p_features)
            Coefficient Matrix
        b : np.ndarray, shape=(n_samples)
            Response variable
        method : str, optional, default='ols-cholesky'
            fitting method, supports one of
            'normal, ols-qr, ols-cholesky'

        Returns
        -------
        np.ndarray
            [description]
        """
        if method == "normal":
            # based on (A'A)^-1 A'b = x
            return np.linalg.inv(A.T @ A) @ A.T @ b
        elif method == "ols-qr":
            # min||(Rx - Q'b)
            q, r = np.linalg.qr(A)
            # solves by forward substitution
            return solve_triangular(r, q.T @ b)
        elif method == "ols-cholesky":
            M = np.linalg.cholesky(A.T @ A)
            y = solve_triangular(M, A.T @ b, lower=True)
            return solve_triangular(M.T, y)

    def fit(
        self, X: np.ndarray, y: np.ndarray, method: str = "ols"
    ) -> LinearRegression:
        """Fits data via ordinary least squares

        Parameters
        ----------
        X : np.ndarray
            design matrix
        y : np.ndarray
            response variable
        method : str, optional
            [description], by default "ols"

        Returns
        -------
        LinearRegression
            [description]
        """
        n_samples, p_features = X.shape[0], X.shape[1]
        X = self.make_polynomial(X)
        self.theta = self._linear_solve(A=X, b=y, method=method)

        # Make the predictions using estimated coefficients
        self.predictions = self.predict(X)
        self.residuals = y - self.predictions
        self.rss = self.residuals @ self.residuals

        # Total parameters fitted
        k = p_features + self.bias
        self.k_params = k

        # Remaining degrees of freedom
        self.ddof = n_samples - k
        self.s2 = self.rss / self.ddof
        ybar = y.mean()
        self.tss = (y - ybar) @ (y - ybar)
        self.ess = self.tss - self.rss
        self.r2 = self.ess / self.tss
        self.bic = n_samples * np.log(self.rss / n_samples) + k * np.log(
            n_samples
        )
        self.run = True

        return self

    def predict(
        self, X: np.ndarray, thetas: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """makes predictions of response variable given input params
        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances
            p_features is number of features
            - if bias is true, a ones column is needed
        thetas:
            if initialized to None:
                uses estimated theta from fitting process
            if array is given:
                makes prediction from given thetas

        Returns:
        predicted values:
            shape = (n_samples,)
        """
        if thetas is None:
            return X @ self.theta
        return X @ thetas

    def _param_covar(self, X: np.ndarray) -> np.ndarray:
        return np.linalg.inv(X.T @ X) * self.s2


class LinearRegressionMLE(LinearBase):
    """
    Implements linear regression via Maximum Likelihood Estimate
    Args:
    bias: indicates if intercept is added or not

    Attributes:
    theta:           Coefficient Weights after fitting
    residuals:       Number of Incorrect Predictions

    Notes:
    Class uses multiple estimation methods to estimate the oridiinary
    lease squares problem min ||Ax - b||, where x = px1, A=nxp, b = nx1
    - A implementation of MLE based on BFGS algorithm is given.  We are
        maximizing log(L(theta)):= L = -n/2 log(2pi *
        residual_std_error**2) - 0.5 ||Ax-b||
        This is same as minimizing 0.5||Ax-b||, the cost function J.
        The jacobian for regression is given by A'(Ax - b) -> (px1) vector
    - A implementation of MLE based on Newton-CG is provided.  The Hessian is:
        A'(Ax - b)A -> pxp matrix
    Todo
    - Levenberg-Marquardt Algorithm

    """

    def __init__(self, bias: bool = True, degree: int = 1):
        self.bias = bias
        self.degree = degree
        self.run = False

    def _loglikelihood(self, true, guess):
        error = true - guess
        return 0.5 * (error ** 2).sum()

    def _objective_func(self, guess: np.ndarray, A: np.ndarray, b: np.ndarray):
        """the objective function to be minimized, returns estimated x for Ax=b
        Args:
        guess:
            initial guess for paramter x
            shape = {1, p_features}
            p_features is the number of columns of design matrix A

        A:
            the coefficient matrix
            shape = {n_samples, n_features}

        b:
            the response variable
            shape = {n_samples, 1}

        Returns:
        Scaler value from loglikelihood function
        """
        y_guess = self.predict(A, thetas=guess)
        f = self._loglikelihood(true=b, guess=y_guess)
        return f

    def _jacobian(self, guess: np.ndarray, A: np.ndarray, b: np.ndarray):
        return A.T @ (guess @ A.T - b)

    def _hessian(self, guess: np.ndarray, A: np.ndarray, b: np.ndarray):
        return A.T @ (A @ guess[:, np.newaxis] - b) @ A

    def _levenberg_marqdt(self):
        raise NotImplementedError()

    def fit(
        self, X: np.ndarray, y: np.ndarray, method: str = "mle_bfgs"
    ) -> LinearRegressionMLE:
        """fits training data via maximum likelihood Estimate

        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y:
            shape = (n_samples)
            Target values

        method:
            the fitting procedure default to 'mle-bfgs'
            Also supports 'mle_newton_cg'

        Returns:
        object
        """
        X = self.make_polynomial(X)
        # generate random guess
        rng = np.random.RandomState(1)
        guess_params = rng.uniform(low=0, high=10, size=X.shape[1])
        if method == "mle_bfgs":
            # doesn't require hessian
            self.theta = minimize(
                self._objective_func,
                guess_params,
                jac=self._jacobian,
                method="BFGS",
                options={"disp": True},
                args=(X, y),
            )
        elif method == "mle_newton_cg":
            # hess is optional but speeds up the iterations
            self.theta = minimize(
                self._objective_func,
                guess_params,
                jac=self._jacobian,
                hess=self._hessian,
                method="Newton-CG",
                options={"disp": True},
                args=(X, y),
            )
        self.predictions = self.predict(X)
        self.run = True
        return self

    def predict(
        self, X: np.ndarray, thetas: Union[np.ndarray, None] = None
    ) -> Union[np.ndarray, Dict]:
        """makes predictions of response variable given input params
        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances
            p_features is number of features
            - if bias is true, a ones column is needed
        thetas:
            if initialized to None:
                uses estimated theta from fitting process
            if array is given:
                it serves as initial guess for optimization

        Returns:
        predicted values:
            shape = (n_samples, 1)
        """
        if thetas is None:
            if isinstance(self.theta, np.ndarray):
                return X @ self.theta
            else:
                return X @ self.theta["x"]
        return X @ thetas


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

    def __init__(
        self,
        eta: float = 0.001,
        n_iter: int = 20,
        random_state: int = 1,
        bias: bool = True,
        degree: int = 1,
    ):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.bias = bias
        self.degree = degree
        self.run = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegressionGD:
        """Fits model to training data via Gradient Descent

        Parameters
        ----------
        X : np.ndarray
            [description]
        y : np.ndarray
            [description]

        Returns
        -------
        LinearRegressionGD
            [description]
        """
        n_samples, p_features = X.shape[0], X.shape[1]
        self.theta = np.zeros(shape=1 + p_features)
        self.cost = []
        X = self.make_polynomial(X)

        for _ in range(self.n_iter):
            # calculate the error
            error = y - self.predict(X)
            self.theta += self.eta * X.T @ error / n_samples
            self.cost.append((error.T @ error) / (2.0 * n_samples))
        self.run = True
        return self

    def predict(
        self, X: np.ndarray, thetas: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        if thetas is None:
            return X @ self.theta
        return X @ thetas
