"""Implementation of coin mixture via EM algorithm"""


from scipy.stats import bernoulli, binom
from scipy.optimize import minimize
from numpy.random import sample
from itertools import chain
import pandas as pd
import numpy as np


class CoinMixture:
    """Class implements the Coin Mixture Model (cmm)

    Experiment consists of randomly picking one coin
    from n_coins and performing m_flips of each coin
    to estimate probability that a particular coin was
    picked
    - Based on following paper: https://www.nature.com/articles/nbt1406

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
        dist = map(lambda x: binom(n, x).pmf(num_heads), p)
        return np.column_stack(tuple(dist))

    def posterior_prob(self, eta: np.ndarray, p: np.ndarray):
        """Computes the posterior probabilities given pmf

        :param eta: pmf of binomial mixture
        :type eta: np.ndarray,
         shape = (n_trials,)
        :param p: probability of success in each trial
        :type prob: np.ndarray
        """
        return eta * p / (eta @ p[:, np.newaxis])

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

        # get count of heads/tail per trial
        m_flips = trial.shape[1]
        num_heads = trial.sum(axis=1)[:, np.newaxis]
        num_tails = m_flips - num_heads

        # get total heads/tails attributed to each coin for m coin flips
        num_heads_per_coin = (gamma * num_heads).sum(axis=0)
        num_tails_per_coin = (gamma * num_tails).sum(axis=0)

        return num_heads_per_coin, num_tails_per_coin

    def mstep(self,
              num_heads_per_coin: np.ndarray,
              num_tails_per_coin: np.ndarray,
              ):
        """[summary]

        :param heads_per_trial: [description]
        :type heads_per_trial: np.ndarray
        :param tails_per_trial: [description]
        :type tails_per_trial: np.ndarray
        :return: [description]
        :rtype: tuple
        """
        return num_heads_per_coin / (num_heads_per_coin + num_tails_per_coin)

    def fit(self,
            trial: np.ndarray, 
            show: bool = False,
            guess: np.ndarray = None,
            ) -> 'CoinMixture':
        """fits a coin mixture model via EM

        :param trial: m_flips each trial
        :type trial: np.ndarray
        :param show: to show the results of iteration, 
         defaults to False
        :type show: bool, optional
        :return: fitted object
        :rtype: CoinMixture
        """
        # intial guess for probabilities associated with each coin
        if guess is None:
            guess = sample(self.n_coins)
        theta = [0] * self.max_iter
        # iterate
        for i in range(self.max_iter):
            theta[i] = chain(guess)
            if show:
                print(f"#{i} ", ", ".join(f"{c:.4f}" for c in chain(guess)))
            
            # compute the e-step
            num_heads, num_tails = self.estep(trial, guess)
            
            # compute the m-step
            guess = self.mstep(num_heads, num_tails)
        
        self.theta = pd.DataFrame(theta)
        return self