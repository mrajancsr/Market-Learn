"""Implementation of Vector AutoRegressive Model"""

from operator import itemgetter
import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import f as ftest
from numpy.linalg import det
from arch.unitroot import PhillipsPerron
from marketlearn.network_causality.vector_ar.varbase import Base
from marketlearn.learning.linear_models.linear_regression import LinearRegression


class BiVariateVar(Base):
    """
    Implementation of bi-variate Vector AutoRegressive Model or order 1

    Note:
    - After a VAR model is specified, granger causality tests can be
    performed
    - Assumes input is log prices whose difference (returns) is stationary
    """
    def __init__(self, fit_intercept: bool = True,
                 degree: int = 1):
        """
        Constructor used to intialize the VAR model

        Currently only supports lag of 1

        :param fit_intercept: (bool) Flag to add bias (True by default)
        :param degree: (int) Lag (1 by default)
        """
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.lr = LinearRegression(fit_intercept=fit_intercept)
        self.run = False
        self.temp_resid = None
        self.lag_order = None
        self.k_params = None
        self.ddof = None
        self.theta = None
        self.predictions = None
        self.residuals = None
        self.design = None
        self.response = None

    def fit(self, x, y, p=1, coint=False) -> 'BiVariateVar':
        """
        Fits the model to training data

        :param x: (np.array) The first variable log returns.
        :param y: (np.array) The second variable in log returns
        :param p: (int) The lagged order
        :return: (object) Class after fitting
        """

        # Create the multivariate response
        Y = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
        n_samples, _ = Y.shape

        if p == 0:
            # Just fit on intercept if any
            Z = np.ones(n_samples)[:, np.newaxis]

        elif p >= 1:
            # Create lagged design matrix and fit intercept if any
            temp = []
            for lag in range(1, p+1):
                z1 = self._shift(Y, num=lag)
                temp.append(z1)
            Z = np.concatenate(temp, axis=1)
            Z = self.make_polynomial(Z)

        # Check for cointegration
        if coint is True:
            self.temp_resid = self.lr.residuals
            # Get the residuals from fitted lineear regression on levels
            resid_lag = self._shift(self.lr.residuals, 1, 0)
            Z = np.concatenate((Z, resid_lag[:, np.newaxis][:-1]), axis=1)

        # Total parameters fitted
        bias = self.fit_intercept
        k = 2*(2*p + bias) if coint is False else 2*(2*p + bias + 1)
        self.lag_order = p
        self.k_params = k
        self.ddof = n_samples - k

        # Compute cholesky decompostion of lagged matrix
        M = np.linalg.cholesky(Z.T @ Z)
        v = solve_triangular(M, Z.T @ Y, lower=True)
        self.theta = solve_triangular(M.T, v).T
        self.predictions = self.predict(Z)
        self.residuals = Y - self.predictions
        self.design = Z
        self.response = Y
        self.run = True

        return self

    @staticmethod
    def _shift(arr: np.ndarray, num: int = 1,
               fill_value: int = 0) -> np.ndarray:
        """
        Shifts a time series by given amount

        :param arr: (np.array) Array to be shifted
        :param num: (int) Number of lag shifts (1 by default)
        :param fill_value: (int) fill value after the shift (0 by defult)
        :return: (np.array) Shifted array
        """
        result = np.empty_like(arr)

        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]

        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]

        else:
            result[:] = arr

        return result

    def predict(self, Z: np.ndarray, thetas: np.ndarray = None) -> np.ndarray:
        """
        Makes predictions of VAR model

        :param Z: (np.array) Lagged matrix of shape (Tx2)
        :param thetas: (np.array) Parameters from fitting with
            shape (2, n+intercept) (None by default)
        :return: (np.array) Predicted values of shape (Tx2)
        """
        if thetas is None:
            return Z @ self.theta.T
        return Z @ thetas.T

    def granger_causality_test(self, alpha=0.05):
        """
        Computes granger causality test on the bivariate VAR model

        :param alpha: (float) Significance level (0.05 by default)
        :return: () *?
        """

        # Get lagged matrix and the two response variables
        idx = self.lag_order + self.fit_intercept
        ydx = range(idx, self.design.shape[1])
        ydx = [0] + list(ydx) if self.fit_intercept else ydx
        xlag = self.design[:, :idx]
        ylag = self.design[:, ydx]
        x = self.response[:, 0]
        y = self.response[:, 1]

        # Regress x against lags of itself
        self.lr.fit_intercept = False
        self.lr.fit(xlag, x)
        xrss_r, xddof_r = self.lr.rss, self.lr.ddof

        # Regress y against lags of itself
        self.lr.fit(ylag, y)
        yrss_r, yddof_r = self.lr.rss, self.lr.ddof

        # Get unstricted rss from original var model
        x_resid = self.residuals[:, 0]
        y_resid = self.residuals[:, 1]
        xrss_u = x_resid @ x_resid
        yrss_u = y_resid @ y_resid
        xddof_u = x_resid.shape[0] - self.k_params / 2
        yddof_u = y_resid.shape[0] - self.k_params / 2

        # Compute F test
        f_stat_x = ((xrss_r - xrss_u) / (xddof_r - xddof_u))
        f_stat_x *= xddof_u / xrss_u
        f_stat_y = (yrss_r - yrss_u) / (yddof_r - yddof_u)
        f_stat_y *= yddof_u / yrss_u

        # Pvalue for Ftest
        x_pval = ftest.cdf(f_stat_x, xddof_r, xddof_u)
        y_pval = ftest.cdf(f_stat_y, yddof_r, yddof_u)

        # Null hypothesis is x does not granger cause y
        result = {}
        result['x_granger_causes_y'] = x_pval < alpha
        result['y_granger_causes_x'] = y_pval < alpha

        return result

    def auto_select(self, series1, series2, lag=5):
        """
        Performs optimal order selection

        :param series1: (np.array) Return series of shape (n_samples,)
        :param series2: (np.array) Return series of shape (n_samples,)
        :param lag: (int) Lag to use
        :return: () *?
        """
        bics = set()
        result = {}
        n = series1.shape[0]

        for p in range(1, lag + 1):
            self.fit(series1, series2, p=p)
            residuals = (self.residuals[:, 0], self.residuals[:, 1])
            resid_cov = np.cov(residuals, ddof=0)
            # Need to check this formula
            bic = np.log(det(resid_cov)) + p * 4 * np.log(n) / n
            bics.add((p, bic))

        result['min_bic'] = min(bics, key=itemgetter(1))
        result['bic_results'] = bics

        return result

    def select_order(self, series1, series2, coint) -> 'BiVariateVar':
        """
        Fits the var model based on auto_select

        :param series1: (np.array) Return series of shape (n_samples,)
        :param series2: (np.array) Return series of shape (n_samples,)
        :param lag: (bool) *?
        :return: (object) Class instance
        """
        result = self.auto_select(series1, series2)
        order = result.get("min_bic")[0]
        self.fit(series1, series2, p=order, coint=coint)

        return self

    @staticmethod
    def _simulate_var(n_samples=1000, corr=0.8):
        """
        Simulates bivariate vector autoregressive model of order 1

        :param n_samples: (int) Number of observations to use (1000 by default)
        :param corr: (float) Correlation between two variables (0.8 by default)
        :return: (tuple) Two vectors with given correlation
        """
        cov_matrix = np.array([[1, corr], [corr, 1]])
        mean_vector = np.zeros(2)
        w = np.random.multivariate_normal(mean=mean_vector,
                                          cov=cov_matrix,
                                          size=n_samples)
        x = np.zeros(n_samples)
        y = np.zeros(n_samples)
        wx = w[:, 0]
        wy = w[:, 1]
        x[0] = wx[0]
        y[0] = wy[0]

        for i in range(1, n_samples):
            x[i] = 3 + 0.4 * x[i-1] + 0.3 * y[i-1] + wx[i]
            y[i] = 5 + 0.2 * x[i-1] + 0.1 * y[i-1] + wy[i]

        return (x, y)

    def coint_test(self, x: np.ndarray, y: np.ndarray, alpha: 0.05) -> bool:
        """Performs Engle Granger co-integration test

        :param x: log price
        :type x: np.ndarray, shape = (n_samples,)
        :param y: log price
        :type y: np.ndarray, shape = (n_samples,)
        :param alpha: significance level
        :type alpha: 0.05
        :return: True if two series are co-integrated
        :rtype: bool
        """
        # Perform a regression of y on x
        self.lr.fit(x[:, np.newaxis], y)

        # Check if residuals are stationary
        pp = PhillipsPerron(self.lr.residuals)

        # Null hypothesis: process is not stationary
        if pp.pvalue < alpha:
            return True
        return False