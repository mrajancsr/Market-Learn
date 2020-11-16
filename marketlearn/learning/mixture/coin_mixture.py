"""Implementation of coin mixture via EM algorithm"""

import pandas as pd
import numpy as np
from itertools import chain
from numpy.random import sample
from scipy.optimize import minimize
from scipy.stats import bernoulli, binom
from string import ascii_uppercase


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

    def _likelihood(self, trial: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Computes the likelihood p(x|z) of each trial

        the function returns P(Evidence | hypothesis)

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
        lik = map(lambda x: binom(n, x).pmf(num_heads), p)
        return np.column_stack(tuple(lik))

    def qprob(self, likelihood: np.ndarray, prior: np.ndarray):
        """Computes the posterior probabilities p(z|x) of each trial

        the function computes the bayes formula;
        p(hypothesis|evidence) = prior * likelihood / p(evidence)
        prior = p(hypothesis) = p(z)
        likelihood = p(x|z)

        :param eta: pmf of binomial mixture
        :type eta: np.ndarray,
         shape = (n_trials,)
        :param p: probability of success in each trial
        :type prob: np.ndarray
        """
        pevidence = likelihood @ prior[:, np.newaxis]
        return likelihood * prior / pevidence

    def estep(self,
              trial: np.ndarray,
              theta: np.ndarray,
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
        psuccess, prior = theta[:2], theta[2:]
        lik = self._likelihood(trial, psuccess)

        # compute the posterior prob of each trial
        return self.qprob(lik, prior)

    def mstep(self,
              trial: np.ndarray,
              qprob: np.ndarray,
              ):
        """Computes the m-step in the EM algorithm

        :param heads_per_trial: [description]
        :type heads_per_trial: np.ndarray
        :param tails_per_trial: [description]
        :type tails_per_trial: np.ndarray
        :return: [description]
        :rtype: tuple
        """
        # get count of heads/tail per trial
        num_heads = trial.sum(axis=1)[:, np.newaxis]

        # estimate pHeads by mle
        eheads = (qprob * num_heads).sum(axis=0)
        pheads = eheads / (qprob.sum(axis=0) * self.m_flips)

        # calculate prior
        prior = qprob.mean(axis=0)
        return np.concatenate((prior, pheads))

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
        # intial guess stores prior and pHeads of each coin
        if guess is None:
            guess = np.concatenate(([0.5, 0.5], sample(self.n_coins)))
        theta = [0] * self.max_iter
        # iterate
        for i in range(self.max_iter):
            theta[i] = chain(guess)
            if show:
                print(f"#{i} ", ", ".join(f"{c:.4f}" for c in chain(guess)))

            # compute the e-step
            qprob = self.estep(trial, guess)

            # compute the m-step
            guess = self.mstep(trial, qprob)

        cols = self._make_titles()
        self.theta = pd.DataFrame(theta, columns=cols)
        return self

    def _make_titles(self) -> list:
        """Creates column titles after em is run

        :return: list of dataframe column titles
        :rtype: list
        """
        # get the letters and create titles
        letters = ascii_uppercase[:self.n_coins]
        col1 = list("p(z={i})".format(i=i) for i in range(self.n_coins))
        col2 = list("theta{i}".format(i=i) for i in letters)
        return list(chain(col1, col2))

    def _flipcoins(self, thetaA, thetaB, m):
        """flips the two coins m times for 5 trials"""
        """trials = 
        np.array([[1,0,0,0,1,1,0,1,0,1], [1,1,1,1,0,1,1,1,1,1], 
        [1,0,1,1,1,1,1,0,1,1], 
        [1,0,1,0,0,0,1,1,0,0],
        [0,1,1,1,0,1,1,1,0,1]])
        """
        m = 100 # 100 tosses of each coin
        theta_A = 0.8
        theta_B = 0.1
        coin_A = bernoulli(theta_A)
        coin_B = bernoulli(theta_B)
        trials = np.vstack([coin_A.rvs(m), coin_A.rvs(m), coin_B.rvs(m), coin_A.rvs(m), coin_B.rvs(m)])
        return trials
