"""Implementation of Linear Regression using various fitting methods"""

import numpy as np
from network_causality.vector_ar.varbase import Base
from scipy.linalg import solve_triangular


class LinearRegression(Base):
    """
    Implements the classic Linear Regression via ols
    Args:
    fit_intercept: indicates if intercept is added or not
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
    lease squares problem min ||Ax - b||, where x = px1 is the paramer
    to be estimated, A=nxp matrix and b = nx1 vector is given
    - A naive implementation of (A'A)^-1 A'b = x is given
      but computing an inverse is expensive
    - A implementation based on QR decomposition is given based on
        min||Ax-b|| = min||Q'(QRx - b)|| = min||(Rx - Q'b)
        based on decomposing nxp matrix A = QR, Q is orthogonal,
        R is upper triangular
    - A cholesky implementation is also included based on converting an n x p
        into a pxp matrix: A'A = A'b, then letting M = A'A & y = A'b, then
        solve Mx = y.  Leting M = U'U, we solve this by forward/backward sub
    """

    def __init__(self, fit_intercept: bool = True, degree: int = 1):
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.run = False

    def estimate_params(self,
                        A: np.ndarray,
                        b: np.ndarray,
                        method: str = 'ols_cholesky') -> np.ndarray:
        """numerically solves Ax = b where x is the parameters to be determined
        based on ||Ax - b||
        Args:
        A:
            coefficient matrix, (n_samples, n_features)
        b:
            target values (n_samples, 1)
        """
        if method == 'normal':
            # based on (A'A)^-1 A'b = x
            return np.linalg.inv(A.T @ A) @ A.T @ b
        elif method == 'ols_qr':
            # min||(Rx - Q'b)
            q, r = np.linalg.qr(A)
            # solves by forward substitution
            return solve_triangular(r, q.T @ b)
        elif method == 'ols_cholesky':
            M = np.linalg.cholesky(A.T @ A)
            y = solve_triangular(M, A.T @ b, lower=True)
            return solve_triangular(M.T, y)

    def fit(self, X:
            np.ndarray,
            y: np.ndarray,
            method: str = 'ols') -> 'LinearRegression':
        """fits training data via ordinary least Squares (ols)
        Args:
        X:
            coefficient matrix, (n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y:
            shape = (n_samples)
            Target values
        covar:
            covariance matrix of fitted parameters i.e theta hat
            set to True if desired
        method:
            the fitting procedure default to cholesky decomposition
            Also supports 'ols_qr' for QR decomposition &
            'normal' for normal equation
        Returns:
        object
        """
        n_samples, p_features = X.shape[0], X.shape[1]
        X = self.make_polynomial(X)
        if method == 'ols-naive':
            self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
        elif method == 'ols':
            M = np.linalg.cholesky(X.T @ X)
            v = solve_triangular(M, X.T @ y, lower=True)
            self.theta = solve_triangular(M.T, v)
        elif method == 'ols-qr':
            # min||(Rx - Q'b)||
            q, r = np.linalg.qr(X)
            # solves by forward substitution
            self.theta = solve_triangular(r, q.T @ y)
        # make the predictions using estimated coefficients
        self.predictions = self.predict(X)
        self.residuals = (y - self.predictions)
        self.rss = self.residuals @ self.residuals
        # total parameters fitted
        k = p_features + self.fit_intercept
        self.k_params = k
        # remaining degrees of freedom
        self.ddof = n_samples - k
        self.s2 = self.rss / self.ddof
        ybar = y.mean()
        self.tss = (y - ybar) @ (y - ybar)
        self.ess = self.tss - self.rss
        self.r2 = self.ess / self.tss
        self.bic = n_samples * np.log(self.rss / n_samples) + \
            k * np.log(n_samples)
        self.run = True
        return self

    def predict(self, X: np.ndarray, thetas: np.ndarray = None) -> np.ndarray:
        """makes predictions of response variable given input params
        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances
            p_features is number of features
            - if fit_intercept is true, a ones column is needed
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
