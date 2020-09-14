"""Implementation of Gaussian Mixture Model
Author: Rajan Subramanian
Created: Sept 13, 2020
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


class GMM:
    """Implements the Gaussian Mixture Model via EM Algorithm"""
    def __init__(self, n_components: int = 2, max_iter=50):
        """Constructor used to specify the number of mixtures

        Currently only supports two gaussian mixtures

        :param n_components: number of gaussian mixtures,
         defaults to 2
        :type n_components: int, optional
        :param max_iter: number of EM iterations to perform,
         defaults to 50
        :type max_iter: int, optional
        """
        self.n_components = n_components
        self.max_iter = max_iter

    def _loglikelihood(self,
                       obs: np.ndarray,
                       means: np.ndarray,
                       sigs: np.ndarray,
                       prob: float,
                       ) -> float:
        """Computes the loglikelihood of GMM

        :param obs: mixture of sample observations
        :type obs: np.ndarray, shape = (n_samples,)
        :param means: mean vector corresponding to n_components
        :type means: np.ndarray, shape = (n_components,)
        :param sigs: vol param corresponding to n_components
        :type sigs: np.ndarray, shape = (n_components,)
        :param prob: prob of observing first gaussian mixture
        :type prob: float
        """
        dist1 = norm(loc=means[0], scale=sigs[0]).pdf(obs)
        dist2 = norm(loc=means[1], scale=sigs[1]).pdf(obs)
        return np.log(dist1 * prob + (1 - prob) * dist2).sum()

    def posterior_density(self, densities, prob):
        n = densities.shape[0]
        d = densities[:, 0]
        d2 = densities[:, 1]
        posterior = (d * prob) / (prob * d + (1 - prob) * d2)
        return posterior

    def norm_density(self, obs, mu1, mu2, sig1, sig2):
        n = len(obs)
        dist1 = norm(loc=mu1, scale=sig1).pdf(obs)
        dist2 = norm(loc=mu2, scale=sig2).pdf(obs)
        densities = np.column_stack((dist1, dist2))
        return densities

    def estep(self, obs, mu1, mu2, sig1, sig2, pi):
        norm_d = self.norm_density(obs, mu1, mu2, sig1, sig2)
        gamma = self.posterior_density(norm_d, pi)
        return gamma

    def mstep(self, obs, gamma):
        mu1 = np.sum(gamma * obs)/np.sum(gamma)
        mu2 = np.sum((1 - gamma)*obs)/np.sum(1-gamma)
        sig1 = np.sqrt(np.sum(gamma * (obs - mu1)**2)/np.sum(gamma))
        sig2 = np.sqrt(np.sum((1 - gamma)*(obs - mu2)**2)/np.sum(1-gamma))
        pi = np.mean(gamma)
        return mu1, mu2, sig1, sig2, pi

    def _pprint(self, c, theta):
        print("#d:\t{:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}".format(c, *theta))

    def fit(self, obs, show=False):
        """fits the GMM model by EM

        :param obs: [description]
        :type obs: [type]
        """
        # initial guess parameters
        mu1, mu2 = obs.mean(), obs.mean()
        sig1, sig2 = obs.std(), obs.std()
        pi = 0.5
        theta = [(mu1, mu2, sig1, sig2, 0.5)]

        #iterate
        for c in range(self.max_iter):
            if show:
                self._pprint(c, theta[c])
            gamma = self.estep(obs, mu1, mu2, sig1, sig2, pi)
            mu1, mu2, sig1, sig2, pi = self.mstep(obs, gamma)
            theta.append((mu1, mu2, sig1, sig2, pi))

        return pd.DataFrame(theta)
    
    def simulate_gmm(self):
        """simulates and fix a gaussian mixture model with two
        components
        """
        # generate mixture of two gaussians thats mixed
        mu_arr = np.array([5, 12])
        sigma_arr = np.array([1.5, 6])
        zi = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
        dist1 = np.random.normal(5, 1.5, 1000)
        dist2 = np.random.normal(12, 6, 1000)
        xs = np.where(zi == 0, dist1, dist2)
        return xs