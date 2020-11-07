"""Module Implements Markov Switching Regression"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from marketlearn.toolz import timethis


class MarkovSwitchingRegression:
    def __init__(self, regime: int = 2):
        """Default constructor used to initialize regime

        Currently only supports mean switching and two regimes

        :param regime: number of regimes, defaults to 2
        :type regime: int, optional
        """
        self.regime = regime
        self.theta = None
        self.tr_matrix = None
        self.filtered_prob = None
        self.predict_prob = None
        self.smoothed_prob = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function

        :param z: intial guess for optimization
        :type z: np.ndarray
        :return: transition probabilities
        :rtype: np.ndarray
        """
        return 1.0 / (1 + np.exp(-z))

    def _normpdf(self,
                 obs: np.ndarray,
                 mean: np.ndarray,
                 sigma: np.ndarray,
                 ) -> np.ndarray:
        """Computes the normal density f(yt|st,Ft-1) of each observation

        :param obs: observed response variable
        :type obs: np.ndarray
        :param mean: means corresponding to 2 regimes
        :type mean: np.ndarray
        :param sigma: volatility corresponding to 2 regimes
        :type sigma: np.ndarray
        :return: normal density of each observation given regime
        :rtype: np.ndarray
        """
        lik = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, sigma)
        return np.column_stack(tuple(lik))

    def _loglikelihood(self,
                       obs: np.ndarray,
                       theta: np.ndarray,
                       store: bool = False,
                       ) -> np.ndarray:
        """Computes the loglikelihood of data observed

        :param obs: observed response variable
        :type obs: np.ndarray
        :param theta: guess for optimization
        :type theta: np.ndarray
        :param store: if true, store results
         of hamilton filter and predicted prob
         defaults to False
        :type store: bool, optional
        :return: loglikelihood as a byproduct of computing
         the optimal forecasts
        :rtype: np.ndarray
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
        # joint density f(yt|St, Ft-1) * P(st|Ft-1)
        jointd = np.zeros(n)

        # construct transition matrix
        self._transition_matrix(*self._sigmoid(prob))

        # initial guess for filter
        hfilter[0] = np.array([0.5, 0.5])
        predict_prob[1] = self.tr_matrix.T @ hfilter[0]

        # compute the densities at t=1
        eta[1] = self._normpdf(obs[1], means, sig)

        # step2: start filter for t = 1,...,T-1
        for t in range(1, n-1):
            exponent = eta[t] * predict_prob[t]
            loglik = exponent.sum()

            # compute the joint density for t = 1,...,T-1
            jointd[t] = loglik
            hfilter[t] = exponent / loglik

            # compute predictions for t = 2,...,T
            predict_prob[t+1] = self.tr_matrix.T @ hfilter[t]
            eta[t+1] = self._normpdf(obs[t+1], means, sig)

        # compute the filter and joint at time T
        exponent = eta[-1] * predict_prob[-1]
        loglik = exponent.sum()
        jointd[-1] = loglik
        hfilter[-1] = exponent / loglik

        if store is True:
            self.filtered_prob = hfilter
            self.predict_prob = predict_prob

        # compute and return loglikelihood of data observed
        return np.log(jointd[1:]).mean()

    def hamilton_filter(self,
                        obs: np.ndarray,
                        theta: np.ndarray,
                        predict: bool = False,
                        ) -> np.ndarray:
        """Computes the hamilton filter

        :param obs: observed response variable
        :type obs: np.ndarray
        :param theta: initial guess parameters
        :type theta: np.ndarray
        :param predict: prediction probabilities
        defaults to False
        :type predict: bool, optional
        :return: filtered probabilities
        :rtype: np.ndarray
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

        # construct transition matrix
        self._transition_matrix(*self._sigmoid(prob))

        # initial guess to start the filter
        hfilter[0] = np.array([0.5, 0.5])
        predict_prob[1] = self.tr_matrix.T @ hfilter[0]

        # compute the densities at t=1
        eta[1] = self._normpdf(obs[1], means, sig)

        # step2: start filter for t =1,..., T-1
        for t in range(1, n-1):
            exponent = eta[t] * predict_prob[t]
            hfilter[t] = exponent / exponent.sum()

            # compute predictions for t = 2,...,T
            predict_prob[t+1] = self.tr_matrix.T @ hfilter[t]
            eta[t+1] = self._normpdf(obs[t+1], means, sig)

        # compute the filter at time T
        exponent = eta[-1] * predict_prob[-1]
        hfilter[-1] = exponent / exponent.sum()

        return hfilter if not predict else (hfilter, predict_prob)

    def _objective_func(self,
                        guess: np.ndarray,
                        obs: np.ndarray,
                        store: bool = False,
                        ) -> float:
        """The objective function to be minimized

        :param guess: parameters for optimization
        :type guess: np.ndarray
        :param obs: observed data
        :type obs: np.ndarray
        :return: negative of loglikelihood of data observed
        :rtype: float
        """
        f = self._loglikelihood(obs, theta=guess, store=store)
        return -f

    def _transition_matrix(self, pii: float, pjj: float):
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
        self.tr_matrix = np.zeros((2, 2))
        self.tr_matrix[0, 0] = pii
        self.tr_matrix[0, 1] = 1 - pii
        self.tr_matrix[1, 1] = pjj
        self.tr_matrix[1, 0] = 1-pjj
        return self

    @timethis
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
                              args=(obs, True))['x']

        # compute the smoothed probabilities
        self.smoothed_prob = \
            self.kims_smoother(self.filtered_prob,
                               self.predict_prob,
                               self.tr_matrix)

        return self

    def kims_smoother(self,
                      filter_prob: np.ndarray,
                      predict_prob: np.ndarray,
                      P: np.ndarray,
                      ) -> np.ndarray:
        """Computes the posterior using full information set

        The posterior p(St=i|FT) is computed via kim's algorithm

        :param filtered_prob: hamilton filter probabilities
        :type filtered_prob: np.ndarray
        :param predict_prob: predicted probabilities
        :type predict_prob: np.ndarray
        :param P: state transition matrix
        :type P: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        n = filter_prob.shape[0]
        smoothed_prob = np.zeros_like(filter_prob)

        # set smoothed prob at time T equal to Hamilton Filter at time T
        smoothed_prob[-1] = filter_prob[-1]

        # recursively compute the smoothed probabilities
        for t in range(n-1, 0, -1):
            terms = (P @ (smoothed_prob[t] / predict_prob[t]))
            smoothed_prob[t-1] = filter_prob[t-1] * terms

        return smoothed_prob

    def _qprob(self,
               filter_prob: np.ndarray,
               predict_prob: np.ndarray,
               P: np.ndarray,
               ) -> np.ndarray:
        """computes the posterior joint probabilities via kim's algorithm

        Posterior joint are given by p(S(t+1)=k, St=i | FT; theta)
        that are computed via
        p(St=i|S(t+1)=k,FT;theta) * p(S(t+1)=k|FT;theta)
        using kim's algorithm

        :param filter_prob: the hamilton filter
         given by p(st=k | Ft) based on info at time t
        :type filter_prob: np.ndarray
        :param predict_prob: prediction probabilities
         given by p(s(t+1)=k | Ft) based on info at time t
        :type predict_prob: np.ndarray
        :param P: transition matrix
        :type P: np.ndarray
        :return: posterior joint probabilities
        :rtype: np.ndarray
        """
        # get the smoothed probabilities
        smooth_prob = self.kims_smoother(filter_prob, predict_prob, P)

        # compute the posterior joint probabilities
        # each observation state is at 00, 01, 10, 11
        n = smooth_prob.shape[0]
        qprob = np.zeros((n, 2**self.regime))

        # -- initial values don't really matter since for
        # -- algorithm, we are starting at index 1

        # calculate the joint, t from t=1
        for t in range(1, n):
            # for state (st-1=0, st=0) and (st-1=0, st=1)
            qprob[t, :2] = \
                P[0] * smooth_prob[t] * filter_prob[t-1, 0] / predict_prob[t]

            # for state (st-1=1, st=0) and (st-1=1, st=1)
            qprob[t, 2:] = \
                P[1] * smooth_prob[t] * filter_prob[t-1, 1] / predict_prob[t]

        # return the full posterior probabilities
        return np.concatenate((smooth_prob, qprob), axis=1)

    def inv_sigmoid(self, x: np.ndarray):
        """computes inverse of sigmoid function

        :param x: the probability
        :type x: np.ndarray
        """
        return -np.log((1-x) / x)

    def _estep(self,
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
        # get hamilton filter and predictions
        hfilter, predict_prob = self.hamilton_filter(obs, theta, predict=True)

        # compute and return posterior prob of each observation
        return self._qprob(filter_prob=hfilter,
                           predict_prob=predict_prob,
                           P=self.tr_matrix)

    def _mstep(self, obs: np.ndarray, qprob: np.ndarray) -> tuple:
        """Computes the m-step in the em algorithm

        :param obs: the actual observations
        :type obs: np.ndarray
        :param qprob: posterior probabilities
        :type qprob: np.ndarray
        :return: poo,p11,mu0,mu1,sig
         which represents the transition prob,
         means for two regimes and constant vol
        :rtype: tuple
        """
        poo = qprob[2:, 2].sum() / qprob[1:, 0].sum()
        p11 = qprob[2:, 4].sum() / qprob[1:, 1].sum()
        mu0 = (qprob[1:, 0] * obs[1:]).sum() / qprob[1: 0].sum()
        mu1 = (qprob[1:, 1] * obs[1:]).sum() / qprob[1: 1].sum()
        spread1, spread2 = obs[1:] - mu0, obs[1:] - mu1
        var = qprob[1:, 0] * spread1**2 + qprob[1:, 1] * spread2**2
        return poo, p11, mu0, mu1, np.sqrt(var.mean())
    
    def run_em_algorithm(self,
                         obs: np.ndarray,
                         show: bool = False,
                         n_iter: int = 20):
        """fits a markov switching model via EM algorithm

        :param obs: initial observations
        :type obs: np.ndarray
        :param show: [description], defaults to False
        :type show: bool, optional
        :param n_iter: [description], defaults to 20
        :type n_iter: int, optional
        """
        pass
