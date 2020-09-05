"""Module Implements Markov Switching Regression"""

import numpy as np
from scipy.linalg import solve_triangular
from sklearn.preprocessing import PolynomialFeatures


class MarkovSwitchingRegression:
    def __init__(self, regime: int = 2, fit_intercept=True):
        self.regime = regime
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function

        :param z: intial guess for optimization
        :type z: np.ndarray
        :return: transition probabilities
        :rtype: np.ndarray
        """
        return 1.0 / (1 + np.exp(-z))

    def _make_polynomial(self, X: np.ndarray) -> np.ndarray:
        bias = self.fit_intercept
        pf = PolynomialFeatures(include_bias=bias)
        return pf.fit_transform(X)

    def _linear_solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Linear solves equation Ax = b

        :param A: design matrix
        :type A: np.ndarray
        :param b: response variable
        :type b: np.ndarray
        :return: x in Ax = b
        :rtype: np.ndarray
        """
        M = np.linalg.cholesky(A.T @ A)
        v = solve_triangular(M, A.T @ b, lower=True)
        return solve_triangular(M.T, v)

    def _beta(self,
              s: int,
              params1: np.ndarray,
              params2: np.ndarray,
              ) -> np.ndarray:
        """Computes the beta of Markov Switching process

        :param s: the state variable, 0 or 1
        :type s: int
        :param params1: parameters corresponding to state 0
        :type params1: np.ndarray, shape = (n_samples,)
        :param params2: parameters corresponding to state 1
        :type params2: np.ndarray
        :raises ValueError: if state is not 0 or 1
        :return: state parameter betas
        :rtype: np.ndarray
        """
        if s not in (0, 1):
            raise ValueError("state regime not supported")
        return params1 if s == 0 else params2

    def _var(self, s: int, var1: float, var2: float) -> float:
        """Returns variance of Markov Switching process

        :param s: state variable, 0 or 1
        :type s: int
        :param var1: variance corresponding to state 0
        :type var1: float
        :param var2: variance corresponding to state 1
        :type var2: float
        :raises ValueError: if state is not 0 or 1
        :return: variance corresponding to regime
        :rtype: float
        """
        if s not in (0, 1):
            raise ValueError("state regime not supported")
        return var1 if s == 0 else var2

    def _normpdf(self,
                 s: int,
                 xt: np.ndarray,
                 yt: np.ndarray,
                 guess: np.ndarray,
                 ) -> np.ndarray:
        """Computes normal density at time t corresponding to regime at state s

        :param s: state variable
        :type s: int
        :param xt: design observation with p features
        :type xt: np.ndarray
        :param yt: response variable at time t
        :type yt: np.ndarray
        :param guess: parameters of msr
         given in following format
         (beta0, beta1, var0, var1)
        :type guess: np.ndarray
        :return: normal density at time t
        :rtype: np.ndarray
        """
        beta = self._beta(s, *guess[0:2])
        var = self._var(s, *guess[2:])
        exponent = (yt - xt.T @ beta) ** 2
        exponent /= (-2.0 * var)
        denom = np.sqrt(2 * np.pi * var)
        return exponent / denom

    def _filtered_probabalities(self,
                                xt: np.ndarray,
                                yt: np.ndarray,
                                theta: np.ndarray,
                                ) -> np.ndarray:
        """calculates the filtered probabilities
           given by p(st=j | Ft)

        :param xt: design matrix
        :type xt: np.ndarray, shape = (n_samples, p_features)
        :param yt: response variable
        :type yt: np.ndarray, shape = (n_samples, 1)
        :param theta: parameters to estimate
         given by (beta0, beta1, var0, var1, p, q)
         where p and q are initial values used in sigmoid function
        :type theta: np.ndarray
        :return: [description]
        :rtype: np.ndarray
        """
        # step 1; initiate starting values
        n_samples = xt.shape[0]
        hfilter = np.zeros((n_samples, self.regime))
        eta_t = np.zeros((n_samples, self.regime))

        # create transition matrix
        pii, pjj = self._sigmoid(theta[-2:])
        P = self._transition_matrix(pii, pjj)

        # linear solve to get starting values of filter
        ones = np.ones(2)[:, np.newaxis]
        A = np.concatenate((ones - P, ones.T))
        b = np.zeros(self.regime + 1)
        b[-1] = 1
        hfilter[0] = self._linear_solve(A, b)

        # compute the density at time 0
        #eta_t[0] = self._normpdf([0, 1], xt, yt, theta[:-2])
        return hfilter


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MarkovSwitchingRegression":
        """fits two state markov-switching

        :param X: design matrix
        :type X: np.ndarray,
         shape = (n_samples, p_features)
        :param y: response variable
        :type y: np.ndarray,
         shape = (n_samples,)
        :return: fitted parameters to data
         param_shape = 2 * (bias + p_features) * k_regimes
        :rtype: MarkovSwitchingRegression
        """
        p_features = X.shape[1]
        X = self._make_polynomial(X)

        # total parameters to be estimated
        # estimate bias, slope, variances for two regimes and transition prob
        bias = self.fit_intercept
        k = 2 * (bias + p_features) * self.regime
        params = np.zeros(k)
        return params
