# pyre-strict
"""
Module Implements Hamilton's Regime Switching Regression
Author: Rajan Subramanian
Date: Nov, 7, 2020

Notes:
- Implementation based on following papers c.f:
  https://econweb.ucsd.edu/~jhamilto/palgrav1.pdf
  https://personal.eur.nl/kole/rsexample.pdf
  https://www.stata.com/features/overview/markov-switching-models/
- Currently, only a two state mean switching
  model with constant variance is supported, i.e

            yt = mu_st + et

  where et~N(0, sigma), mu_st is means corresponding to r.v st
  st ~ Bernoulli(0,1) indicates regime 0 or 1

- EM algorithm is used as initial parameter estimation
  to augment the quasi maximum likelihood estimation
  of joint density function f(yt,st)
  where yt is response variable of observations
  and st is the state transtion variable which
  indicates the regime it came from
  st ~ Bernoulli(0,1) r.v

- todo:
    A) need to refactor and finish
    1) add variance switching
    2) add beta parameter switching
    3) add auto-regressive markov switching model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from numpy import array, float64
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class MarkovSwitchModel:
    """Implementation of Hamilton's Regime Switching Model

    Parameters
    ----------
    nregime : int, optional, default=2
        the number of regimes in the model
    variance_switch : bool, default=False
        whether to perform variance switching
        if False, do mean switching instead

    Attributes
    ----------
    nregime : int
        The number of regimes
    variance_switch : bool
        whether to perform variance switching
    theta : ndarray
        estimated parameters from minimizing loglikelihood
    tr_matrix : ndarray, shape=(n_regimes,nregimes)
        transition matrix
    filtered_prob : ndarray
        hamilton's filter
    predict_prob : ndarray
        predictions
    smoothed_prob : ndarray
        hamilton's smoothed probabilities
    """

    nregime: int = 2
    variance_switch: bool = False
    theta: NDArray = field(init=False)
    tr_matrix: NDArray = field(init=False)
    filtered_prob: NDArray = field(init=False)
    predict_prob: NDArray = field(init=False)
    smoothed_prob: NDArray = field(init=False)
    em_params: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.tr_matrix = np.zeros((self.nregime, self.nregime))
        self.filtered_prob = array([])
        self.predict_prob = array([])
        self.smoothed_prob = array([])
        self.theta = array([])
        self.em_params = pd.Series([])

    def _sigmoid(self, z: ArrayLike[float]) -> ArrayLike[float]:
        """computes the sigmoid function

        Parameters
        ----------
        z : np.ndarray
            initial guess for optimization

        Returns
        -------
        np.ndarray, shape=(n_regime, n_regime)
            transition probabilities
        """
        return 1.0 / (1.0 + np.exp(-z))

    def _normpdf(
        self,
        obs: NDArray[float64],
        mean: NDArray[float64],
        sigma: NDArray[float64],
    ) -> NDArray[float64]:
        """Computes the normal density f(yt|st,Ft-1) of each observation

        Parameters
        ----------
        obs : np.ndarray
            observed response variable
        mean : np.ndarray
            the mean of two regimes
        sigma : np.ndarray
            the volatility of two regimes

        Returns
        -------
        np.ndarray
            normal density given the information set
        """
        lik = map(lambda x, y: norm(loc=x, scale=y).pdf(obs), mean, sigma)
        return np.column_stack(tuple(lik))

    def _loglikelihood(
        self,
        obs: NDArray[float64],
        theta: NDArray[float64],
        store: bool = False,
    ) -> float:
        """Computes the loglikelihood of data observed

        Parameters
        ----------
        obs : np.ndarray
            the observed response variable
        theta : np.ndarray
            initial guess for optimization
        store : bool, optional, default=False
            if true, store results of prediction and filtered probabilities

        Returns
        -------
        np.ndarray
            loglikelihood of data observed
        """
        # get parameters from theta
        prob: NDArray[float64] = theta[:2]
        means = theta[2:4]
        sig = np.array([theta[-1], theta[-1]])

        # step 1: initate starting values
        n = obs.shape[0]
        hfilter = np.zeros((n, self.nregime))
        eta = np.zeros((n, self.nregime))
        predict_prob = np.zeros((n, self.nregime))

        # joint density f(yt|St, Ft-1) * P(st|Ft-1)
        jointd = np.zeros(n)

        pii, pjj = self._sigmoid(prob)
        self._transition_matrix(pii, pjj)

        # initial guess for filter
        hfilter[0] = np.array([0.5, 0.5])
        predict_prob[1] = self.tr_matrix.T @ hfilter[0]

        # compute the densities at t=1:T
        eta = self._normpdf(obs, means, sig)

        # step2: start filter for t = 1,...,T-1
        for t in range(1, n - 1):
            exponent = eta[t] * predict_prob[t]
            loglik = exponent.sum()

            # compute the joint density for t = 1,...,T-1
            jointd[t] = loglik
            hfilter[t] = exponent / loglik

            # compute predictions for t = 2,...,T
            predict_prob[t + 1] = self.tr_matrix.T @ hfilter[t]

        # compute the filter and joint at time T
        exponent = eta[-1] * predict_prob[-1]
        loglik = exponent.sum()
        jointd[-1] = loglik
        hfilter[-1] = exponent / loglik

        if store is True:
            self.filtered_prob = hfilter
            self.predict_prob = predict_prob

        # compute and return loglikelihood of data observed
        return np.log(jointd[1:]).mean()

    def hamilton_filter(
        self,
        obs: NDArray[float64],
        theta: NDArray[float64],
        predict: bool = False,
    ) -> Union[List[NDArray[float64]], NDArray[float64]]:
        """computes the hamilton filter

        Parameters
        ----------
        obs : np.ndarray
            the observations
        theta : np.ndarray
            initial guess for optimization
        predict : bool, optional, default=False
            whether to store result of predicted probabilities

        Returns
        -------
        np.ndarray
            the hamilton filter
        """
        # get parameters from theta
        prob = theta[:2]
        means = theta[2:4]
        sig = np.array([theta[-1], theta[-1]])

        # step 1: initate starting values
        n = obs.shape[0]
        hfilter = np.zeros((n, self.nregime))
        predict_prob = np.zeros((n, self.nregime))

        # construct transition matrix
        self._transition_matrix(*self._sigmoid(prob))

        # initial guess to start the filter
        hfilter[0] = np.array([0.5, 0.5])
        predict_prob[1] = self.tr_matrix.T @ hfilter[0]

        # compute the densities at t=1,..,T
        eta = self._normpdf(obs, means, sig)

        # step2: start filter for t =1,..., T-1
        for t in range(1, n - 1):
            exponent = eta[t] * predict_prob[t]
            hfilter[t] = exponent / exponent.sum()

            # compute predictions for t = 2,...,T
            predict_prob[t + 1] = self.tr_matrix.T @ hfilter[t]

        # compute the filter at time T
        exponent = eta[-1] * predict_prob[-1]
        hfilter[-1] = exponent / exponent.sum()

        return hfilter if not predict else [hfilter, predict_prob]

    def _objective_func(
        self,
        guess: NDArray[float64],
        obs: NDArray[float64],
        store: bool = False,
    ) -> float:
        """the objective function to be minimized

        Parameters
        ----------
        guess : np.ndarray
            initial guess for optimization
        obs : np.ndarray
            observed response variable
        store : bool, optional
            [description], by default False

        Returns
        -------
        float
            negative of loglikelihood to be minimized
        """
        f = self._loglikelihood(obs, theta=guess, store=store)
        return -f

    def _transition_matrix(self, pii: float, pjj: float) -> None:
        """computes transition matrix from state transition probabilities

        Parameters
        ----------
        pii : float
            probability that r.v that starts at i, stays at i
            given by first element of diagonal
        pjj : float
            probability that r.v that starts at j, stays at j
            given by last element of diagonal matrix
        """
        self.tr_matrix[0, 0] = pii
        self.tr_matrix[0, 1] = 1 - pii
        self.tr_matrix[1, 1] = pjj
        self.tr_matrix[1, 0] = 1 - pjj

    def fit(
        self,
        obs: NDArray[float64],
        n_iter: int = 10,
    ) -> MarkovSwitchModel:
        """fits a two state regime switching model

        Parameters
        ----------
        obs : np.ndarray
            [description]
        n_iter : int, optional, default=10
            number of em iterations to perform

        Returns
        -------
        RegimeSwitchModel
            [description]
        """
        # get the initial guess from em algorithm
        self.fit_em(obs, n_iter=n_iter)
        guess_params = self.em_params.tail(1).values.ravel()

        # convert the first two parameters as they are transition probs
        guess_params[:2] = self.inv_sigmoid(guess_params[:2])

        # find optimal parameters
        self.theta = minimize(
            self._objective_func,
            guess_params,
            method="SLSQP",
            options={"disp": True},
            args=(obs, True),
        )["x"]

        # compute the smoothed probabilities with final parameters
        self.smoothed_prob = self.kims_smoother(
            self.filtered_prob, self.predict_prob, self.tr_matrix
        )

        # convert the first two parameters back to transition prob
        self.theta[:2] = self._sigmoid(self.theta[:2])
        return self

    def kims_smoother(
        self,
        filter_prob: NDArray[float64],
        predict_prob: NDArray[float64],
        P: NDArray[float64],
    ) -> NDArray[float64]:
        """Computes the posterior using full information set

        The posterior p(St=i|FT) is computed via kim's algorithm

        Parameters
        ----------
        filter_prob : np.ndarray
            the hamilton filter,
            given by p(st=k | Ft) based on info at time t
        predict_prob : np.ndarray
            the predicted probabilities,
            given by p(s(t+1)=k | Ft) based on info at time t
        P : np.ndarray
            transition matrix

        Returns
        -------
        np.ndarray
            kim's smoothed probabilities
        """
        n = filter_prob.shape[0]
        smoothed_prob = np.zeros_like(filter_prob)

        # set smoothed prob at time T equal to Hamilton Filter at time T
        smoothed_prob[-1] = filter_prob[-1]

        # recursively compute the smoothed probabilities
        for t in range(n - 1, 0, -1):
            terms = P @ (smoothed_prob[t] / predict_prob[t])
            smoothed_prob[t - 1] = filter_prob[t - 1] * terms

        return smoothed_prob

    def _qprob(
        self,
        filter_prob: np.ndarray,
        predict_prob: np.ndarray,
        P: np.ndarray,
    ) -> np.ndarray:
        """computes the posterior joint probabilities via kim's algorithm

        Posterior joint are given by p(S(t+1)=k, St=i | FT; theta)
        that are computed via
        p(St=i|S(t+1)=k,FT;theta) * p(S(t+1)=k|FT;theta)
        using kim's algorithm

        Parameters
        ----------
        filter_prob : np.ndarray
            the hamilton filter,
            given by p(st=k | Ft) based on info at time t
        predict_prob : np.ndarray
            the predicted probability,
            given by p(s(t+1)=k | Ft) based on info at time t
        P : np.ndarray
            transition matrix

        Returns
        -------
        np.ndarray
            [description]
        """
        # get the smoothed probabilities
        smooth_prob = self.kims_smoother(filter_prob, predict_prob, P)

        # compute the posterior joint probabilities
        # each observation state is at 00, 01, 10, 11
        n = smooth_prob.shape[0]
        qprob = np.zeros((n, 2 ** self.nregime))

        # -- initial values don't really matter since for
        # -- algorithm, we are starting at index 1

        # calculate the joint, t from t=1
        for t in range(1, n):
            # for state (st-1=0, st=0) and (st-1=0, st=1)
            qprob[t, :2] = (
                P[0] * smooth_prob[t] * filter_prob[t - 1, 0] / predict_prob[t]
            )

            # for state (st-1=1, st=0) and (st-1=1, st=1)
            qprob[t, 2:] = (
                P[1] * smooth_prob[t] * filter_prob[t - 1, 1] / predict_prob[t]
            )

        # return the full posterior probabilities
        return np.concatenate((smooth_prob, qprob), axis=1)

    def inv_sigmoid(self, x: np.ndarray) -> np.ndarray[np.float32]:
        """computes inverse of sigmoid function

        :param x: the probability
        :type x: np.ndarray
        """
        return -np.log((1 - x) / x)

    def _estep(
        self,
        obs: np.ndarray,
        theta: np.ndarray,
    ) -> np.ndarray:
        """computes the e-step in the EM algorithm

        Parameters
        ----------
        obs : np.ndarray
            the observed response variable
        theta : np.ndarray
            intial guess in EM algorithm

        Returns
        -------
        np.ndarray
            the posterior probabilities computed in the e-step
        """
        # get hamilton filter and predictions
        hfilter, predict_prob = self.hamilton_filter(obs, theta, predict=True)

        # compute and return posterior prob of each observation
        return self._qprob(
            filter_prob=hfilter, predict_prob=predict_prob, P=self.tr_matrix
        )

    # - check to see if more efficient way of summing the means
    def _mstep(
        self, obs: np.ndarray[np.float32], qprob: np.ndarray[np.float32]
    ) -> Tuple[np.ndarray[np.float64], np.ndarray[np.float64], List[float]]:
        """computes m-step in the EM Algorithm

        Parameters
        ----------
        obs : np.ndarray
            the actual observations
        qprob : np.ndarray
            posterior probabilities

        Returns
        -------
        tuple
            poo,p11,mu0,mu1,sig
            which represents the transition prob,
            means for two regimes and constant vol
        """
        poo = qprob[2:, 2].sum() / qprob[1:, 0].sum()
        p11 = qprob[2:, 4].sum() / qprob[1:, 1].sum()
        pkk = np.array([poo, 1 - p11])
        # feels like below step can be done in one shot
        mu0 = (qprob[1:, 0] * obs[1:]).sum() / qprob[1:, 0].sum()
        mu1 = (qprob[1:, 1] * obs[1:]).sum() / qprob[1:, 1].sum()
        muk = np.array([mu0, mu1])
        spread1, spread2 = obs[1:] - mu0, obs[1:] - mu1
        # by default, assume mean switch with constant variance
        if not self.variance_switch:
            var = qprob[1:, 0] * spread1 ** 2 + qprob[1:, 1] * spread2 ** 2
        return pkk, muk, [np.sqrt(var.mean())]

    def fit_em(
        self, obs: np.ndarray, show: bool = False, n_iter: int = 10
    ) -> MarkovSwitchModel:
        """fits a markov switching model via EM Algorithm

        Parameters
        ----------
        obs : np.ndarray
            observations
        show : bool, optional, default=False
            if true, show iterations of EM
        n_iter : int, optional, default=10
            number of EM iterations to perform

        Returns
        -------
        [type]
            [description]
        """
        n = obs.shape[0]
        n_regime = self.nregime

        # pick random index from obs for initial mean estimate
        idx = np.random.randint(low=0, high=n, size=n_regime)

        # initialize means for two regimes, transition probabilites
        muk = obs[idx]
        # ensures the transition probabilities are 0.5 each
        pk = np.zeros(n_regime)
        theta = [0] * n_iter
        sig = np.ones(1)

        # iterate
        for i in range(n_iter):
            theta[i] = np.concatenate((pk, muk, sig))
            if show:
                items = chain(self._sigmoid(pk), muk, sig)
                print(f"#{i} ", ", ".join(f"{c:.4f}" for c in items))

            # compute the e-step
            qprob = self._estep(obs, theta[i])

            # compute the m-step
            pkk, muk, sig = self._mstep(obs, qprob)
            pk = self.inv_sigmoid(pkk)

        cols = self._make_titles()
        self.em_params = pd.DataFrame(theta, columns=cols)
        self.em_params.index.name = "em_iterations"
        self.em_params[["p11", "p22"]] = self.em_params[["p11", "p22"]].apply(
            self._sigmoid
        )
        return self

    def _make_titles(self) -> list:
        """creates column titles after em is run

        Returns
        -------
        list
            list of column titles
        """
        # get the letters and create titles
        n = self.nregime
        col1 = list("p{i}{i}".format(i=i) for i in range(1, n + 1))
        col2 = list("regime{i}_mean".format(i=i) for i in range(1, n + 1))
        col3 = ["regime_vol"]
        return list(chain(col1, col2, col3))
