   """Implementation of coin mixture via EM algorithm"""


import pandas as pd
import numpy as np
from scipy.stats import bernoulli, binom
from scipy.optimize import minimize


class CoinMixture:
    """Class implements the Coin Mixture Model (gmm)

    Currently supports 2 coin mixtures
    """
    def __init__(self,
                 n_components: int = 1,
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
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.theta = None

    def _binom(self, obs: np.ndarray, bias):
        n = obs.shape[0]
        num_heads = obs.count("H")
        return binom(n, bias).pmf(num_heads)

    def estep(self,
             obs: np.ndarray,
             prob: np.ndarray,
             ) -> np.ndarray:
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
        pass

def coin_em(trials, theta_A=None, theta_B=None, maxiter=0):
    # initial guess
    theta_A = theta_A or np.random.random()
    theta_B = theta_B or np.random.random()
    thetas = [(theta_A, theta_B)]
    # iterate
    for c in range(maxiter):
        print("#%d:\t%0.2f %0.2f" % (c, theta_A, theta_B))
        heads_A, tails_A, heads_B, tails_B = e_step(trials, theta_A, theta_B)
        theta_A, theta_B = m_step(heads_A, tails_A, heads_B, tails_B)
        thetas.append((theta_A, theta_B))
    return thetas, (theta_A, theta_B)

def e_step(trials, theta_A, theta_B):
    """produce expected value for  heads_A, heads_B, tails_B
    over the flips given the coin biases
    """
    heads_A, tails_A = 0, 0
    heads_B, tails_B = 0, 0
    for trial in trials:
        likelihood_A = coin_likelihood(trial, theta_A)
        likelihood_B = coin_likelihood(trial, theta_B)
        p_A = likelihood_A / (likelihood_A + likelihood_B)
        p_B = likelihood_B / (likelihood_A + likelihood_B)
        heads_A += p_A * trial.count("H")
        tails_A += p_A * trial.count("T")
        heads_B += p_B * trial.count("H")
        tails_B += p_B * trial.count("T")
    return heads_A, tails_A, heads_B, tails_B

def m_step(heads_A, tails_A, heads_B, tails_B):
    theta_A = heads_A / (heads_A + heads_B)
    theta_B = heads_B / (heads_A + heads_B)
    return theta_A, theta_B

def coin_likelihood(trial, bias):
    numHeads = trial.count("H")
    n = len(trial)
    return binom(n, bias).pmf(numHeads)
