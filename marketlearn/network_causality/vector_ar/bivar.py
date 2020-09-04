"""Implementation of Vector AutoRegressive Model"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import f as ftest
from marketlearn.network_causality.vector_ar.varbase import Base
from marketlearn.learning.linear_models.linear_regression import LinearRegression
from operator import itemgetter
from numpy.linalg import det
from arch.unitroot import PhillipsPerron


class BiVariateVar(Base):
    """Implementation of bi-variate Vector AutoRegressive Model or order 1

    Note:
    - After a VAR model is specified, granger causality tests can be
    performed
    - Assumes input is log prices whose difference (returns) is stationary
    """
    def __init__(self,
                 fit_intercept: bool = True,
                 degree: int = 1,
                 input_type: str = None):
        """Constructor used to intialize the VAR model

        Args:
            Currently only supports lag of 1.
            Defaults to 1.
            fit_intercept (bool, optional): whether to add bias or not.
            Defaults to True.
            degree (int, optional): [description]. Defaults to 1.
            input_type (str, optional): [description]. Defaults to None.
        """
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.lr = LinearRegression(fit_intercept=fit_intercept)
        self.run = False

    def fit(self, x, y, p=1, coint=False) -> 'BiVariateVar':
        """fits the model to training data
        Args:
            x (np.ndarray): the first variable log returns
            y (np.ndarray): the second variable in log returns
            p (int): the lagged order
        Returns:
            object: class after fitting
        """

        # create the multivariate response
        Y = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
        n_samples, k_features = Y.shape
        if p == 0:
            # just fit on intercept if any
            Z = np.ones(n_samples)[:, np.newaxis]
        elif p >= 1:
            # create lagged design matrix and fit intercept if any
            temp = []
            for lag in range(1, p+1):
                z1 = self._shift(Y, num=lag)
                temp.append(z1)
            Z = np.concatenate(temp, axis=1)
            Z = self.make_polynomial(Z)

        # check for cointegration
        if coint is True:
            # get the residuals from fitted lineear regression on levels
            resid_lag = self._shift(self.lr.residuals[:, 0], 1, 0)
            Z = np.concatenate((Z, resid_lag[:, np.newaxis]), axis=1)

        # total parameters fitted
        bias = self.fit_intercept
        k = 2*(2*p + bias) if coint is False else 2*(2*p + bias + 1)
        self.lag_order = p
        self.k_params = k
        self.ddof = n_samples - k
        # compute cholesky decompostion of lagged matrix
        M = np.linalg.cholesky(Z.T @ Z)
        v = solve_triangular(M, Z.T @ Y, lower=True)
        self.theta = solve_triangular(M.T, v).T
        self.predictions = self.predict(Z)
        self.residuals = Y - self.predictions
        self.design = Z
        self.response = Y
        self.run = True
        return self

    def _shift(self,
               arr: np.ndarray,
               num: int = 1,
               fill_value: int = 0) -> np.ndarray:
        """shifts a time series by given amount
        Args:
            arr (np.ndarray): array to be shifted
            num (int, optional): number of lag shifts. Defaults to 1.
            fill_value [None,int], optional): fill value after the shift.
            Defaults to 0.
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
        """makes predictions of VAR model
        Args:
            Z (np.ndarray): lagged matrix, shape = (Tx2)
            thetas (np.ndarray, optional): parameters from fitting.
                                           shape = (2,n+intercept)
                                           Defaults to None.
        Returns:
            np.ndarray: predicted values, shape = (Tx2)
        """
        if thetas is None:
            return Z @ self.theta.T
        return Z @ thetas.T

    def granger_causality_test(self, alpha=0.05, coint=False):
        """Computes granger causality test on the bivariate VAR model"""
        # get lagged matrix and the two response variables
        idx = self.lag_order + self.fit_intercept
        ydx = range(idx, self.design.shape[1])
        ydx = [0] + list(ydx) if self.fit_intercept else ydx
        xlag = self.design[:, :idx]
        ylag = self.design[:, ydx]
        x = self.response[:, 0]
        y = self.response[:, 1]

        # regress x against lags of itself
        self.lr.fit(xlag, x)
        xrss_r, xddof_r = self.lr.rss, self.lr.ddof

        # regress y against lags of itself
        self.lr.fit(ylag, y)
        yrss_r, yddof_r = self.lr.rss, self.lr.ddof

        # get unstricted rss from original var model
        x_resid = self.residuals[:, 0]
        y_resid = self.residuals[:, 1]
        xrss_u = x_resid @ x_resid
        yrss_u = y_resid @ y_resid
        xddof_u = x_resid.shape[0] - self.k_params / 2
        yddof_u = y_resid.shape[0] - self.k_params / 2

        # compute F test
        f_stat_x = ((xrss_r - xrss_u) / (xddof_r - xddof_u))
        f_stat_x *= xddof_u / xrss_u
        f_stat_y = (yrss_r - yrss_u) / (yddof_r - yddof_u)
        f_stat_y *= yddof_u / yrss_u

        # pvalue for Ftest
        x_pval = ftest.cdf(f_stat_x, xddof_r, xddof_u)
        y_pval = ftest.cdf(f_stat_y, yddof_r, yddof_u)

        # null hypothesis is x does not granger cause y
        result = {}
        result['x_granger_causes_y'] = True if x_pval < alpha else False
        result['y_granger_causes_x'] = True if y_pval < alpha else False
        return result

    def auto_select(self, series1, series2, lag=5):
        """performs optimal order selection

        Args:
            series1 ([np.ndarray]): return series
                                    shape = (n_samples,)
            series2 ([type]): return series
                              shape = (n_samples,)
        """
        bics = set()
        result = {}
        n = series1.shape[0]
        for p in range(lag + 1):
            self.fit(series1, series2, p=p)
            residuals = (self.residuals[:, 0], self.residuals[:, 1])
            resid_cov = np.cov(residuals, ddof=0)
            # need to check this formula
            bic = np.log(det(resid_cov)) + p * 4 * np.log(n) / n
            bics.add((p, bic))
        result['min_bic'] = min(bics, key=itemgetter(1))
        result['bic_results'] = bics
        return result

    def select_order(self, series1, series2) -> 'BiVariateVar':
        """fits the var model based on auto_select

        Args:
            series1 (np.ndarray): return series
            series2 (np.ndarray): return series
        Returns:
            object: class instance
        """
        result = self.auto_select(series1, series2)
        order = result.get("min_bic")[0]
        self.fit(series1, series2, p=order)
        return self

    def _simulate_var(self, n_samples=1000, corr=0.8):
        """Simulates bivariate vector autoregressive model of order 1

        Args:
            n_samples (int): Number of observations to use
            Defaults to 1000.
            corr (float): correlation between two variables.
            Defaults to 0.8

        Returns:
            [tuple]: two vectors with given correlation
        """
        cov_matrix = np.array([[1, corr], [corr, 1]])
        mean_vector = np.zeros(2)
        w = np.random.multivariate_normal(
                                         mean=mean_vector,
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
        """performs engle granger cointegration test

        Args:
            x (np.ndarray): log price
            y (np.ndarray): log price

        Returns:
            bool: True if co-integrated, False otherwise
        """
        # perform a regression of y on x
        self.lr.fit(x[:, np.newaxis], y)
        # check if residuals are stationary
        pp = PhillipsPerron(self.residuals)
        # null hypothesis: process is not stationary
        if pp.pvalue < alpha:
            return True
        return False
