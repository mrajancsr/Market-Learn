"""Implementation of Gaussian Mixture Model EM Algorithm"""

from scipy.stats import norm
from itertools import chain
import numpy as np
import pandas as pd

class GaussianMixture:
    """Class implements the Gaussian Mixture Model (gmm)

    Currently supports k-Gaussian mixtures
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
              gamma: np.ndarray,
              ) -> tuple:
        """Computes the m-step in EM algorithm

        :param obs: observed sample of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :param gamma: posterior probabilities
        :type gamma: np.ndarray,
         shape = (n_samples,)
        :return: estimates of means, sigmas and
         state probabilities
        :rtype: tuple
        """
        nk = gamma.sum(axis=0)
        muk = (gamma.T @ obs / nk)[:, np.newaxis]
        vark = (gamma * ((obs - muk)**2).T).sum(axis=0) / nk
        pik = gamma.mean(axis=0)
        return muk.flatten(), np.sqrt(vark), pik

    def fit(self, obs: np.ndarray, show=False) -> 'GaussianMixture':
        """fits a Gaussian Mixture model via EM

        :param obs: observations of mixtures
        :type obs: np.ndarray,
         shape = (n_samples,)
        :return: [description]
        :rtype: GaussianMixture
        """
        n = obs.shape[0]
        n_component = self.n_components

        # pick random index from obs for initial mean estimate
        idx = np.random.randint(low=0, high=n, size=n_component)

        # initialize latent prob, means and sigmas
        pk = np.ones(n_component)
        pk /= pk.sum()
        muk = obs[idx]
        sigk = np.ones(n_component)
        theta = [0] * self.max_iter

        # iterate
        for i in range(self.max_iter):
            theta[i] = chain(muk, sigk, pk)
            if show:
                items = chain(muk, sigk, pk)
                print(f"#{i} ", ", ".join(f"{c:.2f}" for c in items))
            # compute the e-step
            gamma = self.estep(obs, muk, sigk, pk)

            # compute the m-step
            muk, sigk, pk = self.mstep(obs, gamma)

        self.theta = pd.DataFrame(theta)
        return self


