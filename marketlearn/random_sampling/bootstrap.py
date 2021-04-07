"""
Non-parametric estimates of data via sampling with replacement
Author: Rajan Subramanian
Created: July 15, 2020
"""
import numpy as np
import matplotlib.pyplot as plt


class Boot:
    """
    Implements the bootstrap methodology via random sampling with replacement
    in order to quantify uncertainty associated with a estimator


    Attributes:

    Notes:
    Class implements various statistical estimators such as sample mean,
    sample variance, sample covariance, confidence intervals associated with
    these estimates and the standard error via random sampling with replacement
    - A implementation of empirical bootstrap for estimating paramaters
    - A implementation of empirical bootstrap for regression
    - A implementation of residual bootstrap for regression
    - Sequential Bootstrap-todo
    """

    @staticmethod
    def empirical_bootstrap(self, pop: np.ndarray, n=None, B=1000, func=None):
        """Returns the sample statistic from empirical bootstrap method

        Parameters
        ----------
        pop : np.ndarray, shape=(n_samples,)
            the data to sample with replacement
        n : int, optional
            size of the subsample, by default None
        B : int, optional
            the number of boostrap subsamples, by default 1000
        func : Callable, optional
            the statistic we are interested in

        Returns
        -------
        dict
            bootstrapped estimates of sample statistic
        """
        # store the estimates for each bootstrapped sample
        n = pop.shape[0] if n is None else n
        boot_est = [None] * B
        index = 0
        for _ in range(B):
            idx = np.random.randint(low=0, high=n, size=n)
            est = func(pop[idx], axis=0)
            boot_est[index] = est
            index += 1

        result = {}
        result["estimates"] = boot_est
        result["est_mean"] = np.mean(boot_est)
        result["est_err"] = np.std(boot_est, ddof=1)

        return result

    @staticmethod
    def residual_bootstrap(
        self, X: np.ndarray, y: np.ndarray, n=None, B=1000, model=None
    ) -> dict:
        """Computes standard error from regression model using residual bootstrapping

        To be used only if residuals have no hereroscedacity or autocorrelation

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, p_features)
            coefficient matrix
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y : np.ndarray, shape=(n_samples,)
            the response variable
        n : int, optional, default=None
            the size of subsample
        B : int, optional, default=1000
            the number of bootstrapped subsamples
        model : obj, optional, default=None
            regression model object

        Returns
        -------
        dict
            standard error of coefficient estimates
            keys=("estimates", "est_mean", "est_err")
        """
        # fit the model if it hasn't been run
        if model.run is False:
            model.fit(X, y)
        resid = model.residuals
        pred = model.predictions
        boot_est = [None] * B
        result = {}  # to store the mean, std_err
        index = 0
        for _ in range(B):
            idx = np.random.randint(low=0, high=n, size=n)
            boot_yi = pred + resid[idx]
            model.fit(X, boot_yi)
            boot_est[index] = tuple(model.theta)
            index += 1

        # self.boot_est['std_err'] = np.std(statistic, ddof=1, axis=0)
        result["estimates"] = boot_est
        result["est_mean"] = np.mean(boot_est, axis=0)
        result["est_err"] = np.std(boot_est, ddof=1, axis=0)
        return result

    def regression_bootstrap(
        self, X: np.ndarray, y: np.ndarray, n=None, B=1000, model=None
    ):
        """Computes empirical bootstrap for regression problem

        Parameters
        ----------
        X : np.ndarray, (n_samples, p_features)
            design matrix
        y : np.ndarray, (n_samples,)
            the response
        n : int, optional, default=None
            size of the subsample
            if None, use length of data
        B : int, optional, default=1000
            number of bootstrap subsamples
        model : object
            regression model object
        """
        boot_est = [None] * B
        result = {}
        if model.run is False:
            model.fit(X, y)

        thetas = model.theta
        index = 0
        for _ in range(B):
            idx = np.random.randint(low=0, high=n, size=n)
            model.fit(X[idx], y[idx])
            boot_est[index] = tuple(thetas)
            index += 1

        result = {}
        result["estimates"] = boot_est
        result["est_mean"] = np.mean(boot_est, axis=0)
        result["est_err"] = np.std(boot_est, ddof=1, axis=0)

    def plot_hist(self):
        """plots the histogram of estimates"""
        plt.title(f"""Histogram of Sample {self.stat_name}""")
        plt.hist(self.statistic, orientation="horizontal")
