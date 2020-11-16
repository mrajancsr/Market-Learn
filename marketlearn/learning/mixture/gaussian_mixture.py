"""Implementation of Gaussian Mixture Model EM Algorithm"""

import numpy as np
import pandas as pd
from itertools import chain
from scipy.stats import norm
from string import ascii_uppercase


class GaussianMixture:
    """Class implements the Gaussian Mixture Model (gmm)

    Currently supports k-Gaussian mixtures
    - todo convergence until tolerance is reached
    """
    def __init__(self,
                 n_components: int = 1,
                 tol: float = 1e-3,
                 max_iter=100):
        """Default Constructor used to initialize gmm model

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

    def _likelihood(self,
                    obs: np.ndarray,
                    mean: np.ndarray,
                    sigma: np.ndarray,
                    ) -> np.ndarray:
        """Computes the likelihood p(x|z) of each observation

        :param obs: observed sample of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :param mean: mean vector of individual components
        :type mean: np.ndarray,
         shape = (n_components,)
        :param sigma: volatility parameter of individual components
        :type sigma: np.ndarray,
         shape = (n_components,)
        :return: densities of individual mixture components
        :rtype: np.ndarray,
         shape = (n_samples, n_components)
        """
        lik = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, sigma)
        return np.column_stack(tuple(lik))

    def qprob(self, likelihood: np.ndarray, prior: np.ndarray):
        """Computes the posterior probabilities p(z|x) of each observation

        :param eta: densities of Gaussian mixture
        :type eta: np.ndarray,
         shape = (n_samples, n_components)
        :param prob: probability that obversation at index i
         came from density i
        :type prob: float
        """
        pevidence = (likelihood @ prior)[:, np.newaxis]
        return likelihood * prior / pevidence

    def estep(self,
              obs: np.ndarray,
              mean: np.ndarray,
              sigma: np.ndarray,
              prior: np.float64) -> np.ndarray:
        """Computes the e-step in EM algorithm

        :param obs: observed sample of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :param mean: mean vector of each mixtures
        :type mean: np.ndarray,
         shape = (n_components,)
        :param sigma: volatility of each mixture
        :type sigma: np.ndarray,
         shape = (n_components,)
        :param prob: probability that observation at index i
         came from density i
        :type prob: float
        """
        # compute the normal density and posterior prob of each mixture
        likelihood = self._likelihood(obs, mean, sigma)
        return self.qprob(likelihood, prior)

    def mstep(self,
              obs: np.ndarray,
              qprob: np.ndarray,
              ) -> tuple:
        """Computes the m-step in EM algorithm

        :param obs: observed sample of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :param qprob: posterior probabilities
        :type qprob: np.ndarray,
         shape = (n_samples,)
        :return: estimates of means, sigmas and
         prior probabilities
        :rtype: tuple
        """
        # estimate effective number of points assigned to component k
        nk = qprob.sum(axis=0)

        # calculate mean and variance based on mle
        muk = (qprob.T @ obs / nk)[:, np.newaxis]
        vark = (qprob * ((obs - muk)**2).T).sum(axis=0) / nk

        # calculate prior
        prior = nk / qprob.shape[0]
        return prior, muk.flatten(), np.sqrt(vark)

    def fit(self, obs: np.ndarray, show=False) -> 'GaussianMixture':
        """fits a Gaussian Mixture model via EM

        :param obs: observations of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :return: estimated parameters of distribution of latent variables
        :rtype: GaussianMixture
        """
        n = obs.shape[0]
        n_component = self.n_components

        # pick random index from obs for initial mean estimate
        idx = np.random.randint(low=0, high=n, size=n_component)

        # initialize prior, means and sigmas
        pk = np.ones(n_component)
        pk /= pk.sum()
        muk = obs[idx]
        sigk = np.ones(n_component)
        theta = [0] * self.max_iter

        # iterate
        for i in range(self.max_iter):
            theta[i] = chain(pk, muk, sigk)
            if show:
                items = chain(pk, muk, sigk)
                print(f"#{i} ", ", ".join(f"{c:.4f}" for c in items))

            # compute the e-step
            qprob = self.estep(obs, muk, sigk, pk)

            # compute the m-step
            pk, muk, sigk = self.mstep(obs, qprob)

        cols = self._make_titles()
        self.theta = pd.DataFrame(theta, columns=cols)
        return self

    def _make_titles(self) -> list:
        """Creates column titles after em is run

        :return: list of dataframe column titles
        :rtype: list
        """
        # get the letters and create titles
        n = self.n_components
        letters = ascii_uppercase[:n]
        col1 = list("p(z={i})".format(i=i) for i in range(n))
        col2 = list("mean{i}".format(i=i) for i in range(1, n + 1))
        col3 = list("sigma{i}".format(i=i) for i in range(1, n + 1))
        return list(chain(col1, col2, col3))
