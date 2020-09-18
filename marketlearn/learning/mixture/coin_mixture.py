   """Implementation of coin mixture via EM algorithm"""


import pandas as pd
import numpy as np
from scipy.stats import bernoulli, binom
from scipy.optimize import minimize


class CoinMixture:
    """Class implements the Coin Mixture Model (cmm)

    Experiment consists of randomly picking one coin
    from n_coins and performing m_flips of each coin
    to estimate probability that a particular coin was
    picked

    Currently supports 2 coin mixtures
    """
    def __init__(self,
                 n_coins: int = 1,
                 m_flips: int = 10,
                 tol: float = 1e-3,
                 max_iter=100):
        """Default Constructor used to initialize cmm model

        When the constructor is created, theta param is
        initialized to None

        :param tol: convergence threshold.  EM iterations will stop
         when lower bound average gain is below threshold
         defaults to 1e-3
        :type tol: float
        :param n_components: number of mixture components
        :type n_components: int, defaults to 1, optional
        :param max_iter: number of EM iterations to perform,
        :type max_iter: int, defaults to 100, optional
        """
        self.n_coins = n_coins
        self.m_flips = m_flips
        self.tol = tol
        self.max_iter = max_iter
        self.theta = None

    def _binom(self, trial: np.ndarray, p: np.ndarray) -> np.ndarray:
        """computes the probability of successes in each trial

        :param trial: m_flips each trial
        :type trial: np.ndarray,
         shape = (n_trials, m_flips)
         n_trials is total number of trials performed
         m_flips is number of flips per trial
        :param p: probability of successes in each trial
        :type p: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        n = trial.shape[1]
        num_heads = trial.sum(axis=1)
        return binom(n, p).logpmf(num_heads)
    
    def posterior_prob(self, eta: np.ndarray, p: np.ndarray):
        """Computes the posterior probabilities given pmf

        :param eta: pmf of binomial mixture
        :type eta: np.ndarray,
         shape = (n_trials,)
        :param p: probability of success in each trial
        :type prob: np.ndarray
        """
        return eta * p / (eta @ p)

    def estep(self,
              trial: np.ndarray,
              p: np.ndarray,
              ) -> tuple:
        """Computes the e-step in EM algorithm

        :param obs: observed samples of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :param prob: probability that observation at index i
         came from pmf i
        :type prob: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        # compute pmf of each trial
        coin_pmf = self._binom(trial, p)

        # compute the posterior prob of each trial
        gamma = self.posterior_prob(coin_pmf, p)

        # get estimate of count of heads/tail per trial
        m_flips = trial.shape[1]
        heads_per_trial = gamma * trial.sum(axis=1)
        tails_per_trial = gamma * (m_flips - heads_per_trial)

        return (heads_per_trial, tails_per_trial)

    def mstep(self,
              heads_per_trial: np.ndarray,
              tails_per_trial: np.ndarray,
              ):
        """[summary]

        :param heads_per_trial: [description]
        :type heads_per_trial: np.ndarray
        :param tails_per_trial: [description]
        :type tails_per_trial: np.ndarray
        :return: [description]
        :rtype: tuple
        """
        return heads_per_trial / (heads_per_trial + tails_per_trial)

    def fit(self):
        pass