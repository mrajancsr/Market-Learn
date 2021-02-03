"""Module implements a Garch(1,1) model"""

import numpy as np
from scipy.optimize import minimize


class Garch:
    """Class implements Garch(1,1) model
    Notes:
    Conditional Maximum Likelihood Estimation is  used to estimate
    the parameters of Garch Model
    See c.f https://math.berkeley.edu/~btw/thesis4.pdf
    todo - a two point estimation method for medium to large datasets
    """

    def __init__(self, order: tuple = (1, 1), mean: bool = True):
        """Constructor to specify order of Garch and specify mean equation

        :param order: order of Garch Model, defaults to (1, 1)
        :type order: tuple, optional
        :param mean: constant mean to be removed, defaults to True
        :type mean: bool, optional
        """
        self.mean = mean
        self.order = order

    def _loglikelihood(self, vol: np.ndarray, at: np.ndarray) -> float:
        """Returns the loglikelihood function of Garch(1,1) model
            see c.f https://math.berkeley.edu/~btw/thesis4.pdf pg. 19

        :param vol: calculated garch volatility
        :type vol: np.ndarray
        :param at: innovation of return series
        :type at: np.ndarray
        :return: loglikelihood function value
        :rtype: float
        """
        # logliks = 0.5 * (np.log(2*np.pi) + np.log(vol) + a**2/vol)
        # return np.sum(logliks)
        # return 0.5 * (np.log(vol) + a**2 / vol).sum()
        return 0.5 * (np.log(vol * 2 * np.pi) + at ** 2 / vol).sum()

    def _simulate_vol(
        self,
        r: np.ndarray,
        theta: np.ndarray = None,
    ) -> np.ndarray:
        """Simulates the garch(1,1) volatility model
        Args:
            r (np.ndarray): Returns vector
            theta (np.ndarray, optional): estimated weights from fitting.
            Defaults to None. shape = (p_features,)
        Returns:
            [np.ndarray]: predicted volatility
        """
        n = r.shape[0]
        vol = np.zeros(n)
        if theta is None:
            omega, gamma, beta = self.theta
        else:
            omega, gamma, beta = theta
        # set unconditional variance of garch(1,1) as init est
        # vol[0] = omega / (1 - gamma - beta)
        vol[0] = r.var()
        # simulate the garch(1,1) process
        for idx in range(1, n):
            vol[idx] = omega + gamma * r[idx - 1] ** 2 + beta * vol[idx - 1]
        return vol

    def _constraint(self, params: np.ndarray):
        """Specify constraint for variance process
        Args:
            params (np.ndarray): parameters of the optimization
            given by omega, gamma and beta
        """
        # specify the constraint given by gamma + beta < 1
        return 1 - (params[1] + params[2])

    def _objective_func(self, guess: np.ndarray, r: np.ndarray) -> np.float64:
        """Objective function to be minimized
        Args:
            guess (np.ndarray): initial guess for optimization
            r (np.ndarray): the returns vector
        Returns:
            np.float64: value from loglikelihood function
        """
        # get the garch vol
        vol = self._simulate_vol(r=r, theta=guess)
        # estimate the likelihood
        f = self._loglikelihood(vol, at=r)
        return f

    def fit(self, r: np.ndarray) -> Garch:
        """Fits training data via quasi maximum likelihood

        :param r: return series
        :type r: np.ndarray, shape = (n_samples,)
        :return: Garch object with estimated parameters
        :rtype: Garch
        """
        if self.mean:
            a_t = r - r.mean()
        else:
            a_t = r
        # omega, gamma and beta
        guess_params = np.array([a_t.var(), 0.09, 0.90])
        finfo = np.finfo(np.float64)
        bounds = [(finfo.eps, 2 * r.var(ddof=1)), (0.0, 1.0), (0.0, 1.0)]
        cons = {"type": "ineq", "fun": self._constraint}
        self.theta = minimize(
            self._objective_func,
            guess_params,
            method="SLSQP",
            jac=self._jacobian,
            options={"disp": True},
            bounds=bounds,
            args=(a_t),
            constraints=cons,
        )["x"]
        return self

    def _simulate_garch(
        self,
        omega: float = 0.1,
        gamma: float = 0.4,
        beta: float = 0.2,
        size: int = 10000,
    ) -> np.ndarray:
        """Monte-Carlo Simulation of Garch(1,1) model

        :param omega: the first parameter of model
         Defaults to 0.1
        :type omega: float, optional
        :param gamma: the second parameter
         Defaults to 0.4
        :type beta: float, optional
        :param beta: the third paramter
         Defaults to 0.2
        :type beta: float, optional
        :param size: Number of samples
         Defaults to 10000
        :type size: int, optional
        :return: the simulated variance process
        :rtype: [np.ndarray]
        """
        omega = 0.1
        gamma = 0.4
        beta = 0.2

        w = np.random.standard_normal(size)
        a = np.zeros_like(w)
        var = np.zeros_like(w)

        # simulate the garch volatility model
        for t in range(1, size):
            var[t] = omega + gamma * a[t - 1] ** 2 + beta * var[t - 1]
            a[t] = w[t] * np.sqrt(var[t])
        return var

    def _create_beta(self, beta, t):
        def fun(beta, j):
            if j <= 1:
                return 0
            else:
                return beta ** (j - 1)

        return list(map(lambda j: fun(beta, j), range(t)))

    def _jacobian(self, guess: np.ndarray, at: np.ndarray) -> np.ndarray:
        """Computes jacobian of log-likelihood function

        :param guess: initial guess used for optimization
        :type guess: np.ndarray
        :param at: return series
        :type at: np.ndarray
        :return: jacobian of log-likelihood function
        :rtype: np.ndarray
        """
        ht = self._simulate_vol(at, theta=guess)
        integrand = (ht - at ** 2) / (ht ** 2)
        n = ht.shape[0]
        betas = np.cumsum(self._create_beta(guess[-1], n))
        htprime = np.zeros((n, 3))
        for t in range(1, n):
            htprime[t, 0] = betas[t - 1]
            htprime[t, 1] = betas[t - 1] * at[t - 1] ** 2
            htprime[t, 2] = betas[t - 1] * ht[t - 1]
        return 0.5 * (integrand * htprime.T).sum(axis=1)
