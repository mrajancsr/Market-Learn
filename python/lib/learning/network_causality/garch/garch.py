"""Module implements a Garch(1,1) model"""

import numpy as np
from scipy.optimize import minimize


class Garch:
    """Class implements Garch(1,1) model
    Notes:
    Conditional Maximum Likelihood Estimation is  used to estimate
    the parameters of Garch Model
    todo - a two point estimation method for medium to large datasets
    """
    def __init__(self, order: tuple = (1, 1), mean: str = 'const'):
        """Constructor to specify order of Garch and specify mean equation

        Args:
            order (tuple, optional): order of Garch Model
            Defaults to (1, 1)
            mean (str, optional): the mean equation
            Defaults to 'const'
        """
        self.mean = mean
        self.order = order

    def _loglikelihood(self, vol, a):
        """Returns the loglikelihood function of Garch(1,1) model
        Args:
            vol (np.ndarray): the calculated garch volatility
            shape = (n_samples,)
            a (np.ndarray): innovation of returns series
            shape = (n_samples,)
        """
        return 0.5 * (np.log(vol) + a @ a / vol).sum()

    def _simulate_vol(self,
                      r: np.ndarray,
                      theta: np.ndarray = None,
                      ) -> np.ndarray:
        """Simulates the garch(1,1) volatility model
        Args:
            r (np.ndarray): Returns vector
            theta (np.ndarray, optional): estimated weights from fitting.
            Defaults to None. shape = (3,)
        Returns:
            Union[np.ndarray]: predicted volatility
        """
        n = r.shape[0]
        vol = np.zeros(n)
        # use sample variance as initial estimate
        vol[0] = r.var(ddof=1)
        omega, gamma, beta = theta
        for idx in range(1, n):
            vol[idx] = omega + gamma * r[idx-1] + beta * vol[idx-1]
        return vol

        def _constraint(self, params):
            """Specify constraint for variance process
            Args:
                params (np.ndarray): parameters of the optimization
            """
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
        f = self._loglikelihood(vol, a=r)
        return f

    def fit(self, r: np.ndarray) -> 'Garch':
        """Fits training data via quasi maximum likelihood
        Args:
            r (np.ndarray): log returns matrix, shape = (n_samples,)
                            n_samples is number of instances in data
        Returns:
            [object]: Garch estimated parameters
        """
        if self.mean:
            a_t = r - r.mean()
        else:
            a_t = r
        # omega, gamma and beta
        guess_params = np.array([0.1, 0.2, 0.3])
        cons = {'type': 'ineq', 'fun': self._constraint}
        self.theta = minimize(self._objective_func,
                              guess_params,
                              method='BFGS',
                              options={'disp': True},
                              args=(a_t),
                              constraints=cons)['x']
        return self
