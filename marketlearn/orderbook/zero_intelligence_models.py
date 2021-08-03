"""Implementation of Zero-Intelligence Models of Limit Order Book

Author: Rajan Subramanian
Date: 11/09/2020

Notes:
- The following model can be used for price prediction in the
  limit order market assuming agent orders follow poisson arrivals
- Need to finish up the continued fractions when i have time
"""

from functools import wraps
from typing import Callable, Tuple, Union

import matplotlib.pyplot as plt
import mpmath as mpm
import numpy as np
from lmfit import Model
from numpy import floor


class Vectorize:
    """vectorization decorator that works with instance methods"""

    def vectorize(self, otypes=None, signature=None):
        # Decorator as an instance method
        def decorator(fn):
            vectorized = np.vectorize(fn, otypes=otypes, signature=signature)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                return vectorized(*args, **kwargs)

            return wrapper

        return decorator


class LimitOrderBook:
    """Implements a Limit Order Book via array based data structure

    Two Agent Based Models are currently supported:
    - The SFGK Zero Intelligence Model c.f
      https://arxiv.org/pdf/cond-mat/0210475.pdf
    - The Cont-Stoikov-Talreja Model c.f
      https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.1085&rep=rep1&type=pdf

    Currently only supports placing orders of size 1
    """

    # decorator function
    v = Vectorize()

    def __init__(self):
        """Default Constructor used to initialize the order book"""
        self._levels = 1000
        self._depth = 5
        self.price = np.arange(-self._levels, self._levels + 1)
        self.bid_size, self.ask_size = self._initialize_orders()
        self.events = LimitOrderBook._create_events()
        self.run = False

    def _get_levels(self) -> int:
        """returns number of levels in lob

        :return: levels in the lob
        :rtype: int
        """
        return self._levels

    def _get_depth(self) -> int:
        """Returns the depth of orders from the limit order book

        :return: depth of orders in the book
        :rtype: int
        """
        return self._depth

    def _initialize_orders(self) -> Tuple[np.ndarray, np.ndarray]:
        """Creates buy and sell orders in the lob

        Orders are created before agent based simulation begins
        These orders await execution by incoming order or
        can be cancelled

        :return: buy/sell sizes on the bid/ask side
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        d, pl = self._get_depth(), self._get_levels()

        # create buy & sell orders for bid prices at -1000 to -1
        buy_orders = np.repeat(d, pl - 8)
        buy_orders = np.append(buy_orders, [5, 4, 4, 3, 3, 2, 2, 1])
        sell_orders = np.repeat(0, pl)

        # create buy and sell orders for ask prices at 0 to 1000
        buy_orders = np.append(buy_orders, np.repeat(0, pl + 1))
        sell_orders = np.append(sell_orders, [0, 1, 2, 2, 3, 3, 4, 4, 5])
        sell_orders = np.append(sell_orders, np.repeat(d, pl - 8))

        return buy_orders, sell_orders

    # functions to get information from market
    def best_ask(self) -> float:
        """Computes the best price on ask side

        :return: the best asking price is returned
        :rtype: float
        """
        return self.price[self.ask_size > 0].min()

    def best_bid(self) -> float:
        """Computes the best bid on bid side

        :return: the best bid price is returned
        :rtype: float
        """
        return self.price[self.bid_size > 0].max()

    def spread(self) -> float:
        """Compute the inside spread

        :return: spread between best prices
        :rtype: float
        """
        return self.best_ask() - self.best_bid()

    def mid(self) -> float:
        """compute the mid price from best bid and ask

        :return: mid price
        :rtype: float
        """
        return (self.best_ask() + self.best_bid()) * 0.5

    # functions to place order
    def market_buy(self, qty: int = 1):
        """Places market buy order at best ask

        :param qty: the quantity to buy
         if not specified, defaults to 1
        :type qty: int, optional
        """
        # get the best ask and remove orders at that price
        p = self.best_ask()
        remaining_orders = self.ask_size[self.price == p][0] - qty
        self.ask_size[self.price == p] = remaining_orders

    def market_sell(self, qty: int = 1):
        """Places market sell order at best bid

        :param qty: the quantity to sell
         if not specified, defaults to 1.
        :type qty: int, optional

        Places limit buy order at a given price level
        """
        # get best bid and remove orders at that price
        p = self.best_bid()
        remaining_orders = self.bid_size[self.price == p][0] - qty
        self.bid_size[self.price == p] = remaining_orders

    def limit_buy(self, price: int, qty: int = 1) -> None:
        """ "Places limit buy order at a given price level

        Parameters
        ----------
        price : int
            the price at which to place the LBO
        qty : int, optional, default=1
            the quantity agent wishes to buy
        """
        # add orders at the price specified
        new_orders = self.bid_size[self.price == price][0] + qty
        self.bid_size[self.price == price] = new_orders

    def limit_sell(self, price: int, qty: int = 1) -> None:
        """Places limit sell order at a given price level

        Parameters
        ----------
        price : int
            the price at which to place the LSO
        qty : int, optional, default=1
            the quantity agent wishes to sell
        """
        # add orders at the price specified
        new_orders = self.ask_size[self.price == price][0] + qty
        self.ask_size[self.price == price] = new_orders

    def cancel_buy(
        self, price: int, cancelable_orders: int = None, qty: int = 1
    ):
        """Places a cancel order at given price level on bid side

        :param price: price at which we cancel order
         defaults to None.  If not specified, randomly choose
         from which cancelable_orders to cancel from
        :type price: int, optional
        :param cancelable_orders: total cancelable orders on bid side
         defaults to None.  If price is specified,
         then cancelable_orders is None
        :type cancelable_orders: int, optional
        :param qty: the quantity agent wishes to cancel
         defaults to 1
        :type qty: int, optional
        """
        # if price is not specified, raise error
        if price is None:
            raise ValueError("Price cannot be None")

        # otherwise, cancel order on bid side at that price level
        remaining_orders = self.bid_size[self.price == price][0] - qty
        self.bid_size[self.price == price] = remaining_orders

    def cancel_sell(
        self, price: int, cancelable_orders: int = None, qty: int = 1
    ):
        """Places a cancel order at given price level on ask side

        :param price: price at which we cancel order
         defaults to None.  If not specified, randomly choose
         from which cancelable_orders to cancel from
        :type price: int, optional
        :param cancelable_orders: total cancelable orders on ask side
         defaults to None.  If price is specified,
         then cancelable_orders is None
        :type cancelable_orders: int, optional
        :param qty: the quantity agent wishes to cancel
         defaults to 1
        :type qty: int, optional
        """
        # if price is not specified, raise error
        if price is None:
            raise ValueError("Price cannot be None")

        # otherwise, cancel order on ask side at that price level
        remaining_orders = self.ask_size[self.price == price][0] - qty
        self.ask_size[self.price == price] = remaining_orders

    # functions to find the bid/ask positions and mid position
    def _bid_position(self) -> int:
        """returns the index of position of best bid

        :return: index of best bid
        :rtype: int
        """
        return np.where(self.price == self.best_bid())[0][0] + 1

    def _ask_position(self) -> int:
        """returns the index of position of best ask

        :return: index of best ask
        :rtype: int
        """
        return np.where(self.price == self.best_ask())[0][0] + 1

    def _mid_position(self) -> int:
        """returns index of position of mid-market

        :return: index of mid-point price
        :rtype: int
        """
        mid_pos = (self._bid_position() + self._ask_position()) * 0.5
        return int(floor(mid_pos))

    def book_shape(self, band):
        """
        returns the #of shares up to band on each side of book around mid-price
        """
        buy_qty = self.bid_size[self._mid_position() + np.arange(-band - 1, 0)]
        sell_qty = self.ask_size[self._mid_position() + np.arange(band)]
        return np.append(buy_qty, sell_qty)

    def book_plot(self, band: int):
        """plots the book shape around mid-price

        the function plots the number of shares around mid-price
        upto the given band

        :param band: the interval around mid-price to plot
        :type band: int
        """
        shares_around_mid = self.book_shape(band)
        plt.plot(np.arange(-band, band + 1), shares_around_mid)

    def choose(self, price_level: int, prob: np.ndarray = None):
        """pick price from price_level randomly

        :param price_level: the price level from which to pick from
        :type price_level: int
        :param prob: pick a level based on provided prob
         if not specified, pick uniformly.  defaults to None
        :type prob: float, optional
        :return: the price level chosen
        :rtype: int
        """
        # if probability isn't specified, pick uniformly
        if prob is None:
            return np.random.choice(np.arange(1, price_level + 1), 1)[0]
        # otherwise pick based on given probability
        return np.random.choice(np.arange(1, price_level + 1), 1, p=prob)[0]

    @staticmethod
    def _create_events(self) -> dict:
        """creates events based on simulation

        :return: dictionary of events
        :rtype: dict
        """
        events = [
            "market_buy",
            "market_sell",
            "limit_buy",
            "limit_sell",
            "cancel_buy",
            "cancel_sell",
        ]
        return dict(zip(range(6), events))

    def sfgk_model(self, mu: float, lamda: float, theta: float, L: int = 20):
        """Generates an agent based event simulation

        Calling this function generates a market event
        that results in one of market_buy, market_sell,
        limit_buy, limit_sell, cancel_buy, cancel_sell
        according to sfgk model

        :param mu: rate of arrival of market orders
        :type mu: int
        :param lamda: rate of arrival of limit orders
        :type lamda: int
        :param theta: rate at which orders are cancelled
        :type theta: float
        :param L: distance in ticks from opposite best quote
        :type L: int
        """
        # get cancelable orders on bid side from opposte best quote from L
        cancelable_bids = self.price >= (self.best_ask() - L)
        cancelable_bids = self.bid_size[cancelable_bids][:L][::-1]

        # get cancelable orders on ask side from opposite best quote from L
        cancelable_sells = self.price <= (self.best_bid() + L)
        cancelable_sells = self.ask_size[cancelable_sells][-L:]

        # get total orders on bid side from opposite best quote from L
        net_buys = cancelable_bids.sum()

        # get total orders on ask side from opposite best quote from L
        net_sells = cancelable_sells.sum()

        # calculate rates corresponding to cancelable orders
        cb_rates = (theta * cancelable_bids).sum()
        cs_rates = (theta * cancelable_sells).sum()

        # set the probability of each event based on market rates
        cum_rate = mu + 2 * L * lamda + net_buys * theta + net_sells * theta
        pevent = [
            0.5 * mu,
            0.5 * mu,
            L * lamda,
            L * lamda,
            net_buys * theta,
            net_sells * theta,
        ]
        pevent = np.array(pevent) / cum_rate

        # randomly pick a event based on above poisson rates
        market_event = self.events[np.random.choice(6, 1, p=pevent)[0]]

        # generate events
        if market_event == "market_buy":
            self.market_buy()
        elif market_event == "market_sell":
            self.market_sell()
        elif market_event == "limit_buy":
            # pick price from distance L of opposite best quote and place order
            q = self.choose(L)
            p = self.best_ask() - q
            self.limit_buy(price=p)
        elif market_event == "limit_sell":
            # pick price from distance L of opposite best quote and place order
            q = self.choose(L)
            p = self.best_bid() + q
            self.limit_sell(price=p)
        elif market_event == "cancel_buy":
            # calculate rates corresponding to cancelable orders
            cb_rates = (theta * cancelable_bids).sum()

            # determine prob at which orders are cancelled across price levels
            pevent = (theta * cancelable_bids) / cb_rates
            q = self.choose(L, prob=pevent)
            p = self.best_ask() - q
            self.cancel_buy(price=p)
        elif market_event == "cancel_sell":
            # calculate rates corresponding to cancelable orders
            cs_rates = (theta * cancelable_sells).sum()

            # determine prob at which orders are cancelled across price levels
            pevent = (theta * cancelable_sells) / cs_rates
            q = self.choose(L, prob=pevent)
            p = self.best_bid() + q
            self.cancel_sell(price=p)

    def simulate_book(
        self,
        mu: float,
        lamda: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        L: int,
        func: Callable,
        max_events: int = 10000,
    ) -> np.ndarray:
        """Monte-Carlo Simulation of average book shape

        :param mu: rate of arrival of market orders
        :type mu: float
        :param lamda: rate of arrival of limit orders
        :type lamda: Union[float, np.ndarray]
         if func is sfgk_model, float value of lamda is required
         if func is cst_model, array value of lamda is required
        :param theta: rate at which orders are cancelled
        :type theta: Union[float, np.ndarray]
         if func is sfgk_model, float value of theta is required
         if func is cst_model, array values of theta is required
        :param L: distance in ticks from opposite best quote
        :type L: int
        :param func: one of sfgk_model or cst_model
        :type func: Callable
        :param max_events: number of events generated
         defaults to 10000
        :type max_events: int, optional
        :return: average number of orders a distance L ticks
         from opposite best quote
        :rtype: np.ndarray
        """
        # burn-in 1000 events to start
        for _ in range(1000):
            func(mu, lamda, theta, L)

        # calculate average book shape
        avg_book_shape = self.book_shape(L) / max_events

        # run sfgk simulation for max_events and return bookshape
        for _ in range(1, max_events):
            func(mu, lamda, theta, L)
            avg_book_shape += self.book_shape(L) / max_events

        # calculate average of bid/ask orders a distance L ticks
        result = (avg_book_shape[:L][::-1] + avg_book_shape[L + 1 :]) * 0.5
        return np.append(avg_book_shape[L], result)

    def cst_model(self, mu, lamdas, thetas, L):
        """Generates a agent based event simulation

        Calling this function generates a market event
        that results in one of market_buy, market_sell,
        limit_buy, limit_sell, cancel_buy or cancel_sell
        in the cont-stoikov-talreja model

        :param mu: the rate of arrival of market orders
        :type mu: float
        :param lamdas: the rate of arrival of limit orders
        :type lamdas: np.ndarray
        :param thetas: the rate at which orders are cancelled
        :type thetas: np.ndarray
        :param L: [description]
        :type L: [type]
        """
        # get cancelable orders on bid side from opposte best quote from L
        cancelable_bids = self.price >= (self.best_ask() - L)
        cancelable_bids = self.bid_size[cancelable_bids][:L][::-1]

        # get cancelable orders on ask side from opposite best quote from L
        cancelable_sells = self.price <= (self.best_bid() + L)
        cancelable_sells = self.ask_size[cancelable_sells][-L:]

        # get total orders on bid side from opposite best quote from L
        net_buys = cancelable_bids.sum()

        # get total orders on ask side from opposite best quote from L
        net_sells = cancelable_sells.sum()

        # calculate rates corresponding to cancelable orders
        cb_rates = thetas @ cancelable_bids
        cs_rates = thetas @ cancelable_sells

        # calculate total arrival rate of limit orders
        cum_lam = lamdas.sum()

        # set the probability of each event based on market rates
        cum_rate = 2 * mu + 2 * cum_lam + cb_rates + cs_rates
        pevent = np.array([mu, mu, cum_lam, cum_lam, cb_rates, cs_rates])
        pevent /= cum_rate
        market_event = self.events[np.random.choice(6, 1, p=pevent)[0]]

        if market_event == "market_buy":
            self.market_buy()
        elif market_event == "market_sell":
            self.market_sell()
        elif market_event == "limit_buy":
            # pick price from distance L of opposite best quote and place order
            pevent = lamdas / cum_lam
            q = self.choose(L, prob=pevent)
            p = self.best_ask() - q
            self.limit_buy(price=p)
        elif market_event == "limit_sell":
            # pick price from distance L of opposite best quote and place order
            pevent = lamdas / cum_lam
            q = self.choose(L, prob=pevent)
            p = self.best_bid() + q
            self.limit_sell(price=p)
        elif market_event == "cancel_buy":
            # determine prob at which orders are cancelled across price levels
            pevent = (thetas * cancelable_bids) / cb_rates
            q = self.choose(L, prob=pevent)
            p = self.best_ask() - q
            self.cancel_buy(price=p)
        elif market_event == "cancel_sell":
            # determine prob at which orders are cancelled across price levels
            pevent = (thetas * cancelable_sells) / cs_rates
            q = self.choose(L, prob=pevent)
            p = self.best_bid() + q
            self.cancel_sell(price=p)

    def _objective_func(self, k: float, alpha: float, L: float) -> float:
        """Bouchaud et al (2002) power law function

        :param k: power law parameter
        :type k: float
        :param alpha: power law parameter to be estimated
        :type alpha: float
        :param L: distance in ticks from opposite best quote
        :type L: float
        :return: float
        :rtype: float
        """
        return k * L ** (-alpha)

    def _powerlawfit(
        self,
        emp_estimates: np.ndarray,
        L: np.ndarray,
    ) -> np.ndarray:
        """Estimates and returns fitted values from power law function

        :param emp_estimates: empirical arrival rates of limit orders
        :type emp_estimates: np.ndarray
        :param L: distance in ticks from opposite best quote
        :type L: np.ndarray
        :return: arrival rates of limit orders a distance L ticks
         from opposite best quote
        :rtype: np.ndarray
        """
        model = Model(
            self._objective_func,
            independent_vars=["L"],
            param_names=["k", "alpha"],
        )

        # make initial guess for k and alpha
        fit = model.fit(emp_estimates, L=np.arange(1, 6), k=1.2, alpha=0.4)
        return fit.values["k"] * L ** -fit.values["alpha"]

    def set_market_params(
        self,
        mu: float,
        lamda: float,
        theta: float,
    ) -> "Book":
        """sets top of book parameters for market events

        This function needs to be called prior to laplace transformation

        :param mu: market order arrival rates
        :type mu: float
        :param lamda: limit order arrival rates
        :type lamda: float
        :param theta: order cancellation rates
        :type theta: float
        """
        self.mu = mu
        self.lamda = lamda
        self.theta = theta
        # set runner
        self.run = True
        return self

    def _dk(self, k: int):
        """Computes the death rate of birth-death process

        death is defined as an agent's order
        that has either been cancelled or executed
        by incoming market order

        :param mu: market arrival rate
        :type mu: float
        :param k: [description]
        :type k: int
        :param theta: order cancellation rate
        :type theta: float
        """
        if self.run is False:
            raise ValueError("set_market_params needs to be called first")
        return self.mu + k * self.theta

    def _aj(self, k: int, j: int):
        """helper to compute continued fractions

        :param j: index
        :type j: int
        :param lamda: limit order arrival rates
        :type lamda: float
        """
        if self.run is False:
            raise ValueError("set_market_params needs to be called first")
        return -self.lamda * self._dk(k + j - 1)

    def _bj(self, k: int, j: int, s: float):
        """helper to compute continued fractions

        :param k: [description]
        :type k: int
        :param j: [description]
        :type j: int
        :param s: [description]
        :type s: float
        """
        if self.run is False:
            raise ValueError("set_market_params needs to be called first")
        return self.lamda + self._dk(k + j - 1) + s

    @v.vectorize(signature=("(),(),(),()->()"))
    def _laplace_transform(self, k: int, s: float, n: int):
        """Computes the laplace transform of first passage time

        The function evaluates the laplace transform of first
        passage time of agent's orders to go from k orders
        to k-1 orders.  Modified lentz's method is used
        for continued fraction evaluation
        c.f http://turing.ieor.berkeley.edu/html/numerical-recipes/bookcpdf/c5-2.pdf

        :param k: orders
        :type k: int
        :param s: complex number frequency parameter
         s = sigma + i omega, where sigma and omega
         are real numbers
        :type s: float
        :param n: levels of a continued fraction
        :type n: int
        """
        f0 = 10e-30
        c0 = f0
        c = np.zeros(n)
        c[0] = c0
        d = np.zeros(n)
        delta = np.zeros(n)
        f = np.zeros(n)
        f[0] = f0

        # iterate
        for j in range(1, n):
            d[j] = self._bj(k, j, s) + self._aj(k, j) * d[j - 1]
            c[j] = self._bj(k, j, s) + self._aj(k, j) / c[j - 1]
            d[j] = 1.0 / d[j]
            delta[j] = c[j] * d[j]
            f[j] = f[j - 1] * delta[j]

        return -f[-1] / self.lamda

    def fhat(self, b: int, s):
        """laplace transform from state b to state 0

        The function computes and returns the laplace transform
        of the time it takes for agents orders at b
        to deplete to 0

        :param b: [description]
        :type b: int
        :param s: [description]
        :type s: [type]
        """
        orders = np.arange(1, b + 1)
        return self._laplace_transform(b, s, 20).prod()

    def _ghat(self, ask_qty: int, bid_qty: int, s):
        """Laplace transform of probability of increase in mid-price

        The mid-price increases when quantity at ask goes to 0
        before quantity at bid goes to 0.  The function
        computes and returns P(Task_qty < Tbid_qty)
        where Task_qty is first time quantity at ask goes to 0,
        likewise for Tbid_qty

        :param ask_qty: quantity at the best ask price
        :type ask_qty: int
        :param bid_qty: quantity at the best bid price
        :type bid_qty: int
        :param s: complex number frequency parameter
        :type s: [type]
        :return: laplace transform of probability of increase in mid-price
        :rtype: float
        """
        shift = 100
        result = self.fhat(ask_qty, s) * self.fhat(bid_qty, -s) / s
        return result * mpm.exp(-shift * s)

    def prob_mid(self, a, b):
        shift = 100
        f = lambda x: self._ghat(a, b, x)
        return mpm.invertlaplace(f, shift, method="talbot")

    def prob_mid(self, n=10000, xb=1, xs=1):
        """calculates probability of mid-price to go up"""

        def send_orders(xb, xs, mu=0.94, lamda=1.85, theta=0.71):
            cum_rate = 2 * mu + 2 * lamda + theta * xb + theta * xs
            bid_qty_down = mu + theta * xb
            ask_qty_down = mu + theta * xs
            pevent = (
                np.array([lamda, lamda, bid_qty_down, ask_qty_down]) / cum_rate
            )
            ans = np.random.choice(np.arange(4), size=1, p=pevent)[0]

            if ans == 0:
                xb += 1
            elif ans == 1:
                xs += 1
            elif ans == 2:
                xb -= 1
            elif ans == 3:
                xs -= 1
            return xb, xs

        count = 0
        for i in range(n):
            qb_old, qs_old = xb, xs
            while True:
                qb_new, qs_new = send_orders(xb=qb_old, xs=qs_old)
                if qb_new == 0:
                    break
                elif qs_new == 0:
                    count += 1
                    break
                qb_old, qs_old = qb_new, qs_new
        return count / n

    def limit_order_prob(self, n=10000, xb=5, xs=5, dpos=5):
        # assumes my order is first at the bid
        def send_orders(xb, xs, d, mu=0.94, lamda=1.85, theta=0.71):
            cum_rate = 2 * mu + 2 * lamda + theta * (xb - 1) + theta * xs
            ask_qty_down = mu + theta * xs
            pevent = (
                np.array([lamda, lamda, mu, theta * (xb - 1), ask_qty_down])
                / cum_rate
            )
            ans = np.random.choice(np.arange(5), size=1, p=pevent)[
                0
            ]  # pick based on respetive probabilities

            if ans == 0:  # limit buy
                xb += 1
            elif ans == 1:  # limit sell
                xs += 1
            elif ans == 2:  # market sell
                xb -= 1
                d -= 1 if d > 0 else 0
            elif ans == 3:  # cancel buy
                r = np.random.uniform()
                if r > (xb - d) / (xb - 1):
                    d -= 1
                xb -= 1
            else:  # market buy
                xs -= 1
            return xb, xs, d

        count = 0
        for i in range(n):
            qb_old, qs_old, d_old = xb, xs, dpos
            while True:
                qb_new, qs_new, d_new = send_orders(
                    xb=qb_old, xs=qs_old, d=d_old
                )
                if d_new == 0:  # my order has been executed
                    count += 1
                    break
                elif qs_new == 0 and d_new > 0:  # mid price has moved
                    break
                qb_old, qs_old, d_old = qb_new, qs_new, d_new
        return count / n

    def prob_making_spread(self, n=10000, xb=5, xs=5, bid_pos=5, ask_pos=5):
        def send_orders(
            xb, xs, bid_pos, ask_pos, mu=0.94, lamda=1.85, theta=0.71
        ):
            xb_rate = xb - 1
            xs_rate = xs - 1
            if bid_pos == 0:
                xb_rate = xb
            if ask_pos == 0:
                xs_rate = xs

            cum_rate = 2 * mu + 2 * lamda + theta * xb_rate + theta * xs_rate
            pevent = (
                np.array(
                    [lamda, lamda, mu, mu, theta * xb_rate, theta * xs_rate]
                )
                / cum_rate
            )
            ans = np.random.choice(np.arange(6), size=1, p=pevent)[0]

            if ans == 0:
                xb += 1
            elif ans == 1:
                xs += 1
            elif ans == 2:
                xb -= 1
                bid_pos -= 1 if bid_pos > 0 else 0
            elif ans == 3:
                xs -= 1
                ask_pos -= 1 if ask_pos > 0 else 0
            elif ans == 4:
                r = np.random.uniform()
                if r > (xb - bid_pos) / xb_rate:
                    bid_pos -= 1
                xb -= 1
            elif ans == 5:
                r = np.random.uniform()
                if r > (xs - ask_pos) / xs_rate:
                    ask_pos -= 1
                xs -= 1

            return xb, xs, bid_pos, ask_pos

        count = 0
        for i in range(n):
            qb_old, qs_old, b_old, a_old = xb, xs, bid_pos, ask_pos
            while True:
                qb_new, qs_new, b_new, a_new = send_orders(
                    qb_old, qs_old, b_old, a_old
                )
                if b_new == 0 and a_new == 0:
                    count += 1
                    break
                elif qb_new == 0 and a_new > 0:
                    break
                elif qs_new == 0 and b_new > 0:
                    break
                qb_old, qs_old, b_old, a_old = qb_new, qs_new, b_new, a_new
        return count / n
