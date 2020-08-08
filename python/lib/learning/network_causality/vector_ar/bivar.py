"""Implementation of Vector AutoRegressive Model"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import f as ftest
from network_causality.vector_ar.varbase import VarBase


class BiVariateVar(VarBase):
    """Implementation of bi-variate Vector AutoRegressive Model or order 1
    
    Note: 
    - After a VAR model is specified, granger causality tests can be
    performed
    - Assumes input is log prices whose difference (returns) is stationary
    """
    def __init__(self, 
                p: int = 1,
                fit_intercept: bool = True,
                degree: int = 1,
                input_type: str = None):
        """Constructor sets the order, intercept and degree of polynomial
        Args:
            p (int, optional): lag-order. Defaults to 1.
            fit_intercept (bool, optional): bias of the model. Defaults to True.
            degree (int, optional): degree of lagged polynomial. Defaults to 1.
            input_type: (str, optional): input of variables
                                         supports one of log-prices or None
                                         if None, assumes the series is stationary
        """
        self.p = p 
        self.fit_intercept = fit_intercept 
        self.degree = degree
        self.input_type = input_type

    def _convert_inputs(self, z):
        """converts input to returns based on input_type
        Args:
            z ([np.ndarray]): log prices
        Returns:
            [type]: [description]
        """
        if self.input_type == 'log-prices':
            return np.diff(z, axis=0)
        elif self.input_type is None:
            return z
    
    def fit(self, x, y, coint=False) -> BiVariateVar:
        """fits the model to training data
        Args:
            x (np.ndarray): the first variable in log prices
            y (np.ndarray): the second variable in log prices
        Returns:
            BiVariateVar: [description]
        """
        # if co-integrated, build ecm on levels (log prices)
        if coint: 
            self._build_ecm(x, y)
            return self
        # create the multivariate response
        Y = np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)
        # convert the input to desired value
        Y = self._convert_inputs(Y)
        # create lagged design matrix and add intercept
        Z = self._shift(self.make_polynomial(Y))
        # compute cholesky decompostion of lagged matrix
        l = np.linalg.cholesky(Z.T @ Z)
        v = solve_triangular(l, Z.T @ Y, lower=True)
        self.theta = solve_triangular(l.T, v).T
        self.predictions = self.predict(Z)
        self.residuals = Y - self.predictions
        self.design = Z
        self.response = Y
        return self
        
    def _shift(self, arr: np.ndarray, num: int = 1, fill_value: int = 0) -> np.ndarray:
        """shifts a time series by given amount
        Args:
            arr (np.ndarray): array to be shifted
            num (int, optional): number of lag shifts. Defaults to 1.
            fill_value (Any[np.nan, int], optional): fill value after the shift. 
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
    
    def _build_ecm(self, x: np.ndarray,
                   y: np.ndarray, 
                   Y: np.ndarray,
                   Z: np.ndarray):
        """Builds vector error correction model
        Args:
            x (np.ndarray): log prices
            y (np.ndarray): log prices
            Y (np.ndarray): matrix of x and y
            Z (np.ndarray): lagged matrix of x and y
        Returns:
            object: the fitted model
        """
        # if input is not log prices, don't compute ecm
        if self.input_type is None:
            return self 
        else:
            # run regression on levels, x = theta_0 + theta_1 y
            y = self.make_polynomial(y[:, np.newaxis])
            l = np.linalg.cholesky(y.T @ y)
            v = solve_triangular(l, y.T @ x, lower=True)
            level_thetas = solve_triangular(l.T, v)
            # save lagged residuals for VAR model
            residuals = x - y @ level_thetas
            resid_lag = self._shift(residuals[:, 0], 1, 0)
            # build the VAR model
            # create the multivariate response
            Y = np.concatenate((x[:, np.newaxis], y[:, 1]), axis=1)
            # convert the log prices to returns
            Y = self._convert_inputs(Y)
            # create lagged design matrix and add intercept
            Z = self._shift(self.make_polynomial(Y))
            # added lagged residuals from levels to new lagged design matrix
            Z = np.concatenate((Z, resid_lag[:, np.newaxis]), axis=1)
            # compute cholesky decompostion of lagged matrix
            l = np.linalg.cholesky(Z.T @ Z)
            v = solve_triangular(l, Z.T @ Y, lower=True)
            self.theta = solve_triangular(l.T, v).T
            self.predictions = self.predict(Z)
            self.residuals = Y - self.predictions
            self.data = dict(design=Z, response=Y)
            return self
    
    def predict(self, Z: np.ndarray, thetas: Union[np.ndarray, None] = None) -> np.ndarray:
        """makes predictions of VAR model
        Args:
            Z (np.ndarray): lagged matrix, shape = (Tx2)
            thetas (Union[np.ndarray, None], optional): parameters from fitting.
                                                        shape = (2,n+intercept) 
                                                        Defaults to None.
        Returns:
            np.ndarray: predicted values, shape = (Tx2)
        """
        if thetas is None:
            return Z @ self.theta.T
        return Z @ self.theta.T
    
    def granger_causality_test(self, x, y, coint=False):
        """Computes granger causality test on the bivariate VAR model"""
        # null hypothesis is x does not granger cause y
        if self.input_type is None: 
            pass
        else:
            # get lagged values of each variable
            xlag = self._shift(x, 1, 0)
            ylag = self._shift(y, 1, 0)
            # get restricted rss for x and y
            xrss_r, xddof_r = self._lin_regress(x, xlag)
            yrss_r, yddof_r = self._lin_regress(y, ylag)
            # get unstricted rss from original var model
            x_resid = self.residuals[:, 0].flatten()
            y_resid = self.residuals[:, 1].flatten()
            xrss_u = x_resid @ x_resid
            yrss_u = y_resid @ y_resid
            xddof_u = x_resid.shape[0] - (2 + self.intercept)
            yddof_u = y_resid.shape[0] - (2 + self.intercept)
            # compute F test
            f_stat_x = ((xrss_r - xrss_u) / (xddof_r - xddof_u))
            f_stat_x *= xddof_u / xrss_u
            f_stat_y = (yrss_r - yrss_u) / (yddof_r - yddof_u)
            f_stat_y *= yddof_u / yrss_u
            alpha = 0.05
            # pvalue for Ftest
            x_pval = ftest.cdf(f_stat_x, xddof_r, xddof_u)
            y_pval = ftest.cdf(f_stat_y, yddof_r, yddof_u)
            result = {}
            if x_pval < alpha:
                result['x_granger_causes_y'] = True
            else:
                result['x_granger_causes_y'] = False
            if y_pval < alpha: 
                result['y_granger_causes_x'] = True
            else:
                result['y_granger_causes_y'] = False
            return result
            
    def _lin_regress(self, y, x, bias=True):
        """regression of y on x (restricted model)
        Args:
            y ([type]): [description]
            x ([type]): [description]
      
        Returns:
        rss ([float]): residual sum of squares of regression
        """
        n_samples = x.shape[0]
        x = self.make_polynomial(x[:, np.newaxis], fit_intercept=bias)
        l = np.linalg.cholesky(x.T @ x)
        v = solve_triangular(l, x.T @ y, lower=True)
        coefs = solve_triangular(l.T, v)
        pred = x @ coefs
        resid = y - pred
        rss = resid @ resid
        ddof = n_samples - (1 + bias)
        return rss, ddof