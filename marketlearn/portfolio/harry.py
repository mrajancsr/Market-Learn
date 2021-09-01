# pyre-strict
"""
Implementation of Harry Markowitz's Modern Portfolio Theory
Author: Rajan Subramanian
Created Feb 10 2021
"""

from __future__ import annotations

from typing import Dict, Generator, Iterator, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marketlearn.portfolio import Asset
from marketlearn.portfolio.asset import fun
from numpy import array, diag, float64, sqrt
from numpy.random import random
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


class Harry:
    """
    Implements Harry Markowitz'a Model of mean variance optimization
    Parameters
    ----------
    historical_prices : pd.DataFrame
        represents historical daily end of day prices

    Attributes
    ----------
    assets: Dict[name: Asset]
        dictionary of Asset objects whose keys are asset names
    covariance_matrix: np.ndarray
        Scaled Covariance Matrix of daily log returns
    asset_expected_returns: np.ndarray
        forecasted sample means of each Asset
    asset_expected_vol: np.ndarray
        forecasted sample volatility of each Asset
    """

    def __init__(
        self, historical_prices: pd.DataFrame, risk_free_rate: float = 0.01
    ) -> None:
        """Default constructor used to initialize portfolio"""
        self.__assets: Dict[str, Asset] = {
            name: Asset(name=name, price_history=historical_prices[name])
            for name in historical_prices.columns
        }
        self.covariance_matrix: NDArray[float64] = Asset.covariance_matrix(
            tuple(self)
        )
        self.asset_expected_returns: NDArray[
            float64
        ] = self.get_asset_expected_returns()
        self.asset_expected_vol: NDArray[
            float64
        ] = self.get_asset_expected_volatility()

        self.security_count: int = len(self.__assets)
        self.risk_free_rate = risk_free_rate

    def __eq__(self, other: Harry) -> bool:
        return type(self) is type(other) and self.assets() == other.assets()

    def __ne__(self, other: Harry) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        return f"Portfolio size: {self.security_count} Assets"

    def __iter__(self) -> Iterator[Asset]:
        yield from self.assets()

    def assets(self) -> Iterator[Asset]:
        yield from self.__assets.values()

    def get_asset(self, name: str) -> Optional[Asset]:
        """return the Asset in portfolio given name
        Parameters
        ----------
        name : str
            name of the Asset
        Returns
        -------
        Asset
            contains information about the asset
        """
        return self.__assets.get(name)

    def get_asset_expected_returns(self) -> NDArray[float64]:
        """gets expected return of each asset in portfolio
        Yields
        -------
        Iterator[float]
            an iteration of expected return of each asset in portfolio
        """
        return array([asset.expected_returns for asset in self.assets()])

    def get_asset_expected_volatility(self) -> NDArray[float64]:
        """gets expected volatility of each asset in portfolio
        Yields
        -------
        Iterator[float]
            iteration of expected vol of each asset scaled by trading days
        """
        return sqrt(diag(self.covariance_matrix))

    @staticmethod
    def random_weights(
        nsim: int, nsecurities: int
    ) -> Iterator[NDArray[float64]]:
        """creates a portfolio with random weights
        Parameters
        ----------
        nsec : int
            number of securities in porfolio
        nsim : int, optional, default=1
            number of simulations to perform
        Returns
        -------
        np.ndarray, shape=(nsim, nsec)
            random weight matrix
        """
        for _ in range(nsim):
            yield Harry.create_weights(nsecurities)

    @staticmethod
    def create_weights(nsecurities: int) -> NDArray[float64]:
        weights = random(nsecurities)
        return weights / weights.sum()

    def portfolio_variance(self, weights: NDArray[float64]) -> ArrayLike:
        """computes the portfolio variance
        portfolio variance is given by var_p = w'cov(R)w
        Parameters
        ----------
        weights : np.ndarray, shape=(n_assets,), (n_grid, n_assets)
            weight of each asset in portfolio
            n_assets is number of assets in portfolio
            for efficient portfolio, assumes each row
            is a linear combination of grid of two weights
            where one asset is the minimum variance portfolio
            and the other is the markovitz portfolio
        Returns
        -------
        float
            portfolio variance
        """
        return weights.T @ self.covariance_matrix @ weights

    def portfolio_expected_return(
        self, weights: NDArray[float64]
    ) -> ArrayLike:
        return weights.T @ self.asset_expected_returns

    def simulate_investment_opportunity_set(
        self, nportfolios: int
    ) -> Generator[List[float], None, None]:
        """runs a monte-carlo simulation by generating random portfolios

        The random portfolios are generated by first generating random
        weights and computing the standard deviation and expected returns of
        portfolio in each simulation.  The resultant pair of points given by
        (sig_i(p), mu_i(p)) is what is plotted

        Parameters
        ----------
        nsec : int
            [description]
        nportfolios : int, optional
            [description], by default 1
        """
        total_securities = self.security_count
        weights_per_simulation = Harry.random_weights(
            nsim=nportfolios, nsecurities=total_securities
        )

        # return volatility and mean of each simulation as a tuple

        for weights in weights_per_simulation:
            yield [
                np.sqrt(self.portfolio_variance(weights)),
                self.portfolio_expected_return(weights),
            ]

    def graph_simulated_portfolios(self, nportfolios: int) -> None:
        """plots the simulated portfolio
        Parameters
        ----------
        nportfolios : int
            [description]
        """
        simulations = self.simulate_investment_opportunity_set(nportfolios)
        xval, yval = zip(*simulations)

        plt.scatter(xval, yval, marker="o", s=10, cmap="winter", alpha=0.35)

    def sharpe_ratio(self, weights: NDArray[float64]) -> float:
        """Comoputes the sharpe ratio of portfolio given weights
        Parameters
        ----------
        weights : np.ndarray
            percentage of each asset held in portfolio
        Returns
        -------
        sharpe ratio
            float
        """
        return (
            self.portfolio_expected_return(weights) - self.risk_free_rate
        ) / np.sqrt(self.portfolio_variance(weights))

    def optimize_risk(
        self,
        add_constraints: bool = False,
        target: Optional[float] = None,
        bounds: Optional[Tuple[float, float]] = None,
    ) -> NDArray[float64]:
        """Computes the weights corresponding to minimum variance

        Parameters
        ----------
        add_constraints : bool, optional, default=False
            mean constraint
        target : float, optional, default=None
            target portfolio return if constraints is True
        bounds: Tuple[float, float], default = (0.0, 1.0)
            sets the range for each weights in portfolio
            (0.0, 1.0) is long only position
            (-1.0, 0) is short only
            (-1, 1) allows for both long and short positions

        Returns
        -------
        np.ndarray[float64]
            weights corresponding to minimum variance
        """
        total_assets = self.security_count
        guess_weights = Harry.random_weights(nsim=1, nsecurities=total_assets)

        # minimize risk subject to target level of return constraint
        if add_constraints:
            constraints = [
                {
                    "type": "eq",
                    "fun": lambda w: self.portfolio_expected_return(w)
                    - target,
                },
                {"type": "eq", "fun": lambda w: sum(w) - 1},
            ]
        # minimize risk with only contraint sum of weights = 1
        else:
            constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1}]

        # minimize the portolio variance, assume no short selling
        weights = minimize(
            fun=lambda w: self.portfolio_variance(w),
            x0=guess_weights,
            constraints=constraints,
            method="SLSQP",
            bounds=bounds,
        )["x"]
        return weights

    def optimize_sharpe(
        self, bounds: Optional[Tuple[int]] = None
    ) -> NDArray[float64]:
        """Return the weights corresponding to maximizing sharpe ratio
        The sharpe ratio is given by SRp = mu_p - mu_f / sigma_p
        subject to w'mu = mu_p
                   w'cov(R)w = var_p
                   s.t sum(weights) = 1

        Parameters
        ----------
        bounds : Optional[Tuple[int]], default=None
            bounds for long/short positins
            (0, 1) long only
            (-1, 0) short only
            (-1, 1) can be long or short

        Returns
        -------
        NDArray[float64]
            weights corresponding to optimal sharpe ratio
        """
        total_assets = self.security_count
        guess_weights = Harry.random_weights(nsim=1, nsecurities=total_assets)

        # set target return & target variance and sum of weights constraint
        consts = [{"type": "eq", "fun": lambda w: sum(w) - 1}]

        # maximize sharpe subject to above constraints
        weights = minimize(
            fun=lambda w: -1 * self.sharpe_ratio(w),
            x0=guess_weights,
            constraints=consts,
            method="SLSQP",
            bounds=bounds,
        )["x"]

        return weights

    def construct_efficient_frontier(
        self, bounds: Optional[Tuple[float, float]] = None
    ) -> Tuple[NDArray[float64], NDArray[float64]]:
        """Constructs the efficient frontier

        Parameters
        ----------
        bounds : [type], optional, default=None
            bound for each asset weight
            for long position, bound is (0, 1)
            for short position, bound is (-1, 0)

        Returns
        -------
        tuple
            efficient portfolio's volatility and expected returns
        """
        # get minimum variance portfolio
        m = self.optimize_risk(bounds=bounds)

        # compute mean of global minimum variance portfolio
        minimum_var_portfolio_mean = self.portfolio_expected_return(m)

        # get efficient portfolio x whose target return is max security returns
        target = self.asset_expected_returns.max()
        x = self.optimize_risk(
            add_constraints=True, target=target, bounds=bounds
        )

        # compute grid of values
        theta = np.linspace(-1, 1, 1000)

        # portfolio z is linear combination of above two portfolios
        z = theta[:, np.newaxis] * m + (1 - theta[:, np.newaxis]) * x

        # compute portfolio mean and variance with above weights
        efficient_portfolio_mean = self.portfolio_expected_return(z)
        efficient_portfolio_var = self.portfolio_variance(z)

        # get the max volatility of assets in universe
        max_vol = self.asset_expected_vol.max()

        # return pair of mean returns and vol for new efficient portfolio
        mu_p = efficient_portfolio_mean[
            efficient_portfolio_mean >= minimum_var_portfolio_mean
        ]
        sig_p = sqrt(
            efficient_portfolio_var[
                efficient_portfolio_mean >= minimum_var_portfolio_mean
            ]
        )

        return sig_p[sig_p <= max_vol], mu_p[sig_p <= max_vol]

    def graph_frontier(
        self, nportfolios: int, bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        """graphs the efficient frontier set

        Parameters
        ----------
        nportfolios : int
            [description]
        bounds : [type], optional
            [description], by default None
        """
        plt.figure(figsize=(8, 5))
        self.graph_simulated_portfolios(nportfolios)

        # construct the frontier
        xval, yval = self.construct_efficient_frontier(bounds=bounds)
        plt.plot(xval, yval, color="orange", linewidth=4)
        plt.grid()

    def graph_investment_opportunity_set(
        self, bounds: Optional[Tuple[float, float]] = None
    ) -> None:
        # create grid of target returns > minimum variance portfolio mean
        target_returns = np.linspace(0, 1, 1000)
        # todo: this can be made faster
        weights = map(
            lambda x: self.optimize_risk(
                add_constraints=True, target=x, bounds=bounds
            ),
            target_returns,
        )
        weights = np.vstack(tuple(weights))

        # get expected returns and vols
        portfolio_expected_returns = self.portfolio_expected_return(weights)
        portfolio_expected_vol = np.sqrt(self.portfolio_variance(weights))

        # plot graph of sig vs expected returns
        plt.figure(figsize=(8, 5))
        plt.plot(
            portfolio_expected_vol,
            portfolio_expected_returns,
            color="black",
            linestyle="-",
        )
        plt.grid()


test = fun()
print(test)
