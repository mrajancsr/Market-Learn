"""Module Implements Markov Switching Regression"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class MarkovSwitchingRegression:
    def __init__(self, regime: int = 2):
        self.regime = regime
        self.theta = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function

        :param z: intial guess for optimization
        :type z: np.ndarray
        :return: transition probabilities
        :rtype: np.ndarray
        """
        return 1.0 / (1 + np.exp(-z))

    def _likelihood(self,
                    obs: np.ndarray,
                    mean: np.ndarray,
                    sigma: np.ndarray,
                    ):
        """Computes the likelihood f(yt|st,Ft-1) of each observation

        :param obs: [description]
        :type obs: np.ndarray
        :param beta: [description]
        :type beta: np.ndarray
        :param sigma: [description]
        :type sigma: np.ndarray
        :return: [description]
        :rtype: [type]
        """
        lik = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, sigma)
        return np.column_stack(tuple(lik))

    def _loglikelihood(self, obs, theta: np.ndarray):
        """computes the loglikelihood of data observed"""
        # get parameters from theta
        prob = theta[:2]
        means = theta[2:4]
        sig = np.array([theta[-1], theta[-1]])

        # step 1: initate starting values
        n = obs.shape[0]
        hfilter = np.zeros((n, self.regime))
        eta = np.zeros((n, self.regime))
        predictions = np.zeros((n, self.regime))
        # joint density f(yt|St, Ft-1) * P(st|Ft-1)
        jointd = np.zeros(n)

        # create transition matrix
        pii, pjj = self._sigmoid(prob)
        P = self._transition_matrix(pii, pjj)

        # initial guess for filter
        hfilter[0] = np.array([0.5, 0.5])
        predictions[1] = P.T @ hfilter[0]

        # compute the densities at t=1
        eta[1] = self._likelihood(obs[1], means, sig)
        self.n = n

        # step2: start filter for t = 1,...,T-1
        for t in range(1, n-1):
            exponent = eta[t] * predictions[t]
            loglik = exponent.sum()

            # compute the joint density for t = 1,...,T-1
            jointd[t] = loglik
            hfilter[t] = exponent / loglik

            # compute predictiosn for t = 2,...,T
            predictions[t+1] = P.T @ hfilter[t]
            eta[t+1] = self._likelihood(obs[t+1], means, sig)

        # compute the filter and joint at time T
        exponent = eta[-1] * predictions[-1]
        loglik = exponent.sum()
        jointd[-1] = loglik
        hfilter[-1] = exponent / loglik

        # compute and return loglikelihood of data observed
        return np.log(jointd[1:]).mean()

    def hamilton_filter(self,
                        obs: np.ndarray,
                        theta: np.ndarray,
                        predict=False):
        """computes the hamilton filter, p(st=j|Ft)

        :param obs: observed response variable
        :type obs: np.ndarray,
         shape = (n_samples,)
        :param theta: parameters of density function
        :type theta: np.ndarray
        :param predict: the prediction probabilities
         if True, return prediction probabilities
         along with the Hamilton Filter as tuple
         defaults to False
        :type predict: bool, optional
        :return: hamilton filter
        :rtype: np.ndarray if predict is False
         otherwise a tuple of np.ndarrays
        """
        # get parameters from theta
        prob = theta[:2]
        means = theta[2:4]
        sig = np.array([theta[-1], theta[-1]])

        # step 1: initate starting values
        n = obs.shape[0]
        hfilter = np.zeros((n, self.regime))
        eta = np.zeros((n, self.regime))
        predictions = np.zeros((n, self.regime))

        # create transition matrix
        pii, pjj = self._sigmoid(prob)
        P = self._transition_matrix(pii, pjj)

        # initial guess to start the filter
        hfilter[0] = np.array([0.5, 0.5])
        predictions[1] = P.T @ hfilter[0]

        # compute the densities at t=1
        eta[1] = self._likelihood(obs[1], means, sig)

        # step2: start filter for t =1,..., T-1
        for t in range(1, n-1):
            exponent = eta[t] * predictions[t]
            hfilter[t] = exponent / exponent.sum()
            predictions[t+1] = P.T @ hfilter[t]
            eta[t+1] = self._likelihood(obs[t+1], means, sig)

        # compute the filter at time T
        exponent = eta[-1] * predictions[-1]
        hfilter[-1] = exponent / exponent.sum()
        return hfilter if not predict else (hfilter, predictions)

    def _objective_func(self, guess: np.ndarray, obs: np.ndarray) -> float:
        """The objective function to be minimized

        :param guess: parameters for optimization
        :type guess: np.ndarray
        :param obs: observed data
        :type obs: np.ndarray
        :return: negative of loglikelihood of data observed
        :rtype: float
        """
        f = self._loglikelihood(obs, theta=guess)
        return -f

    def _transition_matrix(self, pii: float, pjj: float) -> np.ndarray:
        """Constructs the transition matrix given the diagonal probabilities

        :param pii: probability that r.v
         stays at state i given it starts at i
         given by first element of diagonal
        :type pii: float
        :param pjj: probability that r.v
         stays at state j given it starts at j
         given by next element of diagonal
        :type pjj: float
        :return: transition matrix
        :rtype: np.ndarray
        """
        transition_matrix = np.zeros((2, 2))
        transition_matrix[0, 0] = pii
        transition_matrix[0, 1] = 1 - pii
        transition_matrix[1, 1] = pjj
        transition_matrix[1, 0] = 1-pjj
        return transition_matrix

    def fit(self, obs: np.ndarray) -> "MarkovSwitchingRegression":
        """Fits two state markov switching model

        :return: parameters from optimization
        :rtype: object
        """
        guess_params = np.array([0.5, 0.5, 4, 10, 2.0])
        self.theta = minimize(self._objective_func,
                              guess_params,
                              method='BFGS',
                              options={'disp': True},
                              args=(obs,))['x']

        # compute the hamilton filter from the minimization step
        self.filtered_prob = self.hamilton_filter(obs, self.theta)

        return self

    def smoothing_prob(self,
                       transition_matrix: np.ndarray,
                       hfilter: np.ndarray) -> np.ndarray:
        """Computes smoothing probabilities via kim's algorithm

        :param transition_matrix: computed from initial guess
        :type transition_matrix: np.ndarray
        :param hfilter: [description]
        :type hfilter: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        
