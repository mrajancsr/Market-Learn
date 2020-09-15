import seaborn as sns
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy.stats import norm


class GaussianMixture:
    """Class implements the Gaussian Mixture Model

    Currently only supports a two mixture model
    """
    def __init__(self,
                 n_components: int = 2,
                 tol: float = 1e-3,
                 max_iter=100):
        """Default Constructor used to initialize gmm model

        :param tol: convergence threshold.  EM iterations will stop
         when lower bound average gain is below threshold
         defaults to 1e-3
        :type tol: float
        :param n_components: number of mixture components
        :type n_components: int, defaults to 2, optional
        :param max_iter: number of EM iterations to perform,
        :type max_iter: int, defaults to 100, optional
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def _initialize_params(self) -> "GaussianMixture":
        """Initializes parameters of GMM based on n_components

        Creates the initial values for mean vector, covariance matrix
        of the mixture and the state probability vector, p(zi = j | xi, theta)

        :return: object with default parameters
        :rtype: GaussianMixture
        """
        n = self.n_components
        self.mean_arr = np.zeros(n)
        self.sigma_arr = np.ones(n)
        self.prob = np.array([0.5, 0.5])
        return self

    def _norm(self,
              obs: np.ndarray,
              mean: np.ndarray,
              sigma: np.ndarray,
              ) -> np.ndarray:
        """Constructs univariate normal densities

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
        dist = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, sigma)
        return np.column_stack(tuple(dist))

    def posterior_prob(self, eta: np.ndarray, prob: np.ndarray):
        """Computes the posterior probabilities given densities

        :param eta: densities of Gaussian mixture
        :type eta: np.ndarray,
         shape = (n_samples, n_components)
        :param prob: probability that obversation at index i
         came from density i
        :type prob: float
        """
        return eta * prob / (eta @ prob)[:, np.newaxis]

    def estep(self,
              obs: np.ndarray,
              mean: np.ndarray,
              sigma: np.ndarray,
              prob: float):
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
        normal_densities = self._norm(obs, mean, sigma)
        gamma = self.posterior_prob(normal_densities, prob)
        return gamma

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
        pass

def gmm_em(trial, mu1, mu2, sigma1, sigma2, pi, maxiter=0, show=False):
    theta = [(mu1, mu2, sigma1, sigma2, pi)]
    # iterate
    for c in range(maxiter):
        if show:
            print("#%d:\t%0.2f %0.2f %0.2f %0.2f %0.2f" % (c, mu1, mu2, sigma1, sigma2, pi))
        gamma = gmm_estep(trial, mu1, mu2, sigma1, sigma2, pi)
        mu1, mu2, sigma1, sigma2, pi = gmm_mstep(trial, gamma)
        theta.append((mu1, mu2, sigma1, sigma2, pi))
    return theta, (mu1, mu2, sigma1, sigma2, pi)


def norm_likelihood(trial, mu1, mu2, sigma1, sigma2, pi):
    dist1 = norm(loc=mu1, scale=sigma1).pdf(trial)
    dist2 = norm(loc=mu2, scale=sigma2).pdf(trial)
    return np.log(dist1 * pi + (1-pi)*dist2).sum()

def normal_density(trial, mu1, mu2, sigma1, sigma2):
    n = len(trial)
    dist1 = norm(loc=mu1, scale=sigma1).pdf(trial)
    dist2 = norm(loc=mu2, scale=sigma2).pdf(trial)
    densities = np.column_stack((dist1, dist2))
    return densities

def posterior_density(densities, pi):
    n = densities.shape[0]
    d = densities[:, 0]
    d2 = densities[:, 1]
    posterior = np.zeros(n)
    posterior = (d * pi) / (pi * d + (1 - pi) * d2)
    return posterior

def gmm_estep(trial, mu1, mu2, sigma1, sigma2, pi):
    norm_d = normal_density(trial, mu1, mu2, sigma1, sigma2)
    gamma = posterior_density(norm_d, pi)
    return gamma

def gmm_mstep(trial, gamma):
    n = trial.shape[0]
    mu1 = np.sum(gamma * trial)/np.sum(gamma)
    mu2 = np.sum((1 - gamma)*trial)/np.sum(1-gamma)
    sig1 = np.sqrt(np.sum(gamma * (trial - mu1)**2)/np.sum(gamma))
    sig2 = np.sqrt(np.sum((1 - gamma)*(trial - mu2)**2)/np.sum(1-gamma))
    pi = np.mean(gamma)
    return mu1, mu2, sig1, sig2, pi

def simulate_gaussian(show=False, maxiter=15):
    X = np.random.multivariate_normal([-5, 10], [[0.5, 0], [0, 9]], 1000)
    zi = np.random.choice([0, 1], size=1000, p=[0.75, 0.25])
    xs = np.where(zi == 0, X[:, 0], X[:, 1])
    n = len(xs)
    idx = np.random.randint(low=0, high=n, size=2)
    theta, _ = gmm_em(xs, xs[idx[0]], xs[idx[1]], 1, 1, 0.5, maxiter=maxiter, show=show)

    return pd.DataFrame(theta), idx, xs, X