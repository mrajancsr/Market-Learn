"""Module Implements Markov Switching Regression"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures


class MarkovSwitchingRegression:
    def __init__(self, degree: int = 1, regime: int = 2, fit_intercept=True):
        self.degree = degree
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
        degree = self.degree
        pf = PolynomialFeatures(degree=degree, include_bias=bias)
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
                 xt: np.ndarray,
                 yt: np.ndarray,
                 beta: np.ndarray,
                 sig: float,
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

        zt = yt - xt @ beta
        exponent = np.exp(-(zt ** 2) / (2 * sig**2))
        denom = np.sqrt(2 * np.pi) * sig
        return exponent / denom

    def _loglikelihood(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       theta: np.ndarray,
                       ) -> float:
        """returns loglikelihood of two state markov
           switching model

        :param X: design matrix
        :type X: np.ndarray, shape = (n_samples, p_features)
        :param y: response variable
        :type y: np.ndarray, shape = (n_samples,)
        :param thetas: parameters of msr given by:
         (beta00, beta01, beta10, beta11, var0, var1, p, q)
         (beta00 & beta10 are intercepts)
         (beta01 & beta11 are slopes)
         (var0 & var1 are regime variances)
        :type theta: np.ndarray
        :return: log-likelihood function value
        :rtype: float
        """
        # get parameters from theta
        intercept, slope = theta[:2], theta[2:4]
        sig = theta[4:6]
        prob = theta[6:]

        # step 1; initiate starting values
        n_samples = X.shape[0]
        hfilter = np.zeros((n_samples, self.regime))
        eta = np.zeros((n_samples, self.regime))
        cond_density = np.zeros(n_samples)
        predictions = np.zeros((n_samples, self.regime))

        # create transition matrix
        pii, pjj = self._sigmoid(prob)
        P = self._transition_matrix(pii, pjj)

        # linear solve to start the filter
        ones = np.ones(2)[:, np.newaxis]
        A = np.concatenate((ones - P, ones.T))
        b = np.zeros(self.regime + 1)
        b[-1] = 1
        hfilter[0] = self._linear_solve(A, b)
        predictions[0] = P @ hfilter[0]

        # compute the densities for two regimes at time 0
        eta[0] = self._normpdf(X[0], y[0], [intercept, slope], sig)

        # step2: start the filter
        for t in range(1, n_samples):
            exponent = predictions[t-1] * eta[t-1]
            loglik = exponent.sum()
            cond_density[t] = loglik
            hfilter[t] = exponent / loglik
            predictions[t] = P @ hfilter[t]
            eta[t] = self._normpdf(X[t], y[t], [intercept, slope], sig)

        # step3: calculate the loglikelihood, ignore index 0
        return np.log(cond_density[1:]).mean()

    def hamilton_filter(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        predict: bool = False,
                        ) -> np.ndarray:
        """Computes the Hamilton Filter, p(st=j|Ft)

        Assumes self.fit() has been called prior to computing
        the filter.  If fit() is not called, error is raised

        :param X: design_matrix
        :type X: np.ndarray, shape = (n_samples, p_features)
         n_samples is number of observations
         p_features is number of features
        :param y: response variable
        :type y: np.ndarray, shape = (n_samples,)
        :param predict: if True return prediction
         probabilities along with the filter as a tuple,
         defaults to False
        :type predict: bool, optional
        :raises AttributeError: if fit() has not been called
        :return: Filtered probabilities
        :rtype: np.ndarray
        """
        if self.theta is None:
            raise AttributeError("fit() has not been called")

        # get parameters from theta
        intercept, slope = self.theta[:2], self.theta[2:4]
        sig = self.theta[4:6]
        prob = self.theta[6:]

        # step 1; initiate starting values
        X = self._make_polynomial(X)
        n_samples = X.shape[0]
        hfilter = np.zeros((n_samples, self.regime))
        eta = np.zeros((n_samples, self.regime))
        cond_density = np.zeros(n_samples)
        predictions = np.zeros((n_samples, self.regime))

        # create transition matrix
        pii, pjj = self._sigmoid(prob)
        P = self._transition_matrix(pii, pjj)

        # linear solve to start the filter
        ones = np.ones(2)[:, np.newaxis]
        A = np.concatenate((ones - P, ones.T))
        b = np.zeros(self.regime + 1)
        b[-1] = 1
        hfilter[0] = self._linear_solve(A, b)
        predictions[0] = P @ hfilter[0]

        # compute the densities for two regimes at time 0
        eta[0] = self._normpdf(X[0], y[0], [intercept, slope], sig)

        # step2: start the filter
        for t in range(1, n_samples):
            exponent = predictions[t-1] * eta[t-1]
            loglik = exponent.sum()
            cond_density[t] = loglik
            hfilter[t] = exponent / loglik
            predictions[t] = P @ hfilter[t]
            eta[t] = self._normpdf(X[t], y[t], [intercept, slope], sig)

        return hfilter if predict is True else (hfilter, predictions)

    def _objective_func(self,
                        guess: np.ndarray,
                        X: np.ndarray,
                        y: np.ndarray):
        """the objective function to be minimized

        :param guess: parameters for optimization
        :type guess: np.ndarray
        :param X: design matrix
        :type X: np.ndarray
        :param y: response variable
        :type y: np.ndarray
        :return: scaler value from minimization
        :rtype: np.float64
        """
        f = self._loglikelihood(X, y, theta=guess)
        return -f

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
        X = self._make_polynomial(X)

        # total parameters to be estimated
        # estimate bias, slope, variances for two regimes and transition prob
        # k = 2 * (bias + p_features) * self.regime
        guess_params = np.array([0.05, 0.01, 0.2, 0.4, y.std(), y.std(), 0.5, 0.5])
        self.theta = minimize(self._objective_func,
                              guess_params,
                              method='SLSQP',
                              options={'disp': True},
                              args=(X, y))['x']

        return self
