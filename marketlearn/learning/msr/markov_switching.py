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

            # compute predictions for t = 2,...,T
            predictions[t+1] = P.T @ hfilter[t]
            eta[t+1] = self._likelihood(obs[t+1], means, sig)

        # compute the filter and joint at time T
        exponent = eta[-1] * predictions[-1]
        loglik = exponent.sum()
        jointd[-1] = loglik
        hfilter[-1] = exponent / loglik

        # compute and return loglikelihood of data observed
        return np.log(jointd[1:]).mean()

    def switching_probs(self,
                        obs: np.ndarray,
                        theta: np.ndarray,
                        smoothing: bool = False):
        """Computes the Hamilton filter, predictions and smoothing prob

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
        predict_prob = np.zeros((n, self.regime))

        # create transition matrix
        pii, pjj = self._sigmoid(prob)
        P = self._transition_matrix(pii, pjj)

        # initial guess to start the filter
        hfilter[0] = np.array([0.5, 0.5])
        predict_prob[1] = P.T @ hfilter[0]

        # compute the densities at t=1
        eta[1] = self._likelihood(obs[1], means, sig)

        # step2: start filter for t =1,..., T-1
        for t in range(1, n-1):
            exponent = eta[t] * predict_prob[t]
            hfilter[t] = exponent / exponent.sum()

            # compute predictions for t = 2,...,T
            predict_prob[t+1] = P.T @ hfilter[t]
            eta[t+1] = self._likelihood(obs[t+1], means, sig)

        # compute the filter at time T
        exponent = eta[-1] * predict_prob[-1]
        hfilter[-1] = exponent / exponent.sum()

        # set smoothed prob at time T equal to Hamilton Filter at time T
        if smoothing is True:
            smoothed_prob = np.zeros_like(hfilter)
            smoothed_prob[-1] = hfilter[-1]

            # recursively compute the smoothed probabilities
            for t in range(n-1, 0, -1):
                terms = (P @ (smoothed_prob[t] / predict_prob[t]))
                smoothed_prob[t-1] = hfilter[t-1] * terms

        if not smoothing:
            return (hfilter, predict_prob, P)
        else:
            return (hfilter, predict_prob, smoothed_prob, P)

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

        # get all probabilities and transition matrix from minimization step
        self.filtered_prob, self.predict_prob, \
            self.smoothed_prob, self.transition_matrix = \
            self.switching_probs(obs, self.theta, smoothing=True)

        return self

    def qprob(self,
              filtered_prob: np.ndarray,
              predict_prob: np.ndarray,
              P: np.ndarray,
              ) -> np.ndarray:
        """Computes the posterior probabilities

        The posterior is simply the smoothing probability
        computed via kim's algorithm

        :param filtered_prob: hamilton filter probabilities
        :type filtered_prob: np.ndarray
        :param predict_prob: predicted probabilities
        :type predict_prob: np.ndarray
        :param P: state transition matrix
        :type P: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        n = filtered_prob.shape[0]
        smoothed_prob = np.zeros_like(filtered_prob)

        # set smoothed prob at time T equal to Hamilton Filter at time T
        smoothed_prob[-1] = filtered_prob[-1]

        # recursively compute the smoothed probabilities
        for t in range(n-1, 0, -1):
            terms = (P @ (smoothed_prob[t] / predict_prob[t]))
            smoothed_prob[t-1] = filtered_prob[t-1] * terms

        return smoothed_prob

    def estep(self,
              obs: np.ndarray,
              theta: np.ndarray,
              ) -> np.ndarray:
        """Computes the e-step in EM algorithm

        :param obs: observed response variables
        :type obs: np.ndarray
        :param theta: parameters to be estimated
        :type theta: np.ndarray
        :return: posterior probabilities
        :rtype: np.ndarray
        """
        # get the switching parameters from theta
        filtered_prob, predict_prob, P = self.switching_probs(obs, theta)

        # compute the posterior prob of each observation
        return self.qprob(filtered_prob, predict_prob, P)

    def mstep(self, obs: np.ndarray, qprob: np.ndarray):
        """Computes the m-step in the EM algorithm

        :param obs: [description]
        :type obs: np.ndarray
        :param qprob: [description]
        :type qprob: np.ndarray
        """
        pass
