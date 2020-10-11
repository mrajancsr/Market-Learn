"""Implementation of Zero-Intelligence Models of Limit Order Book"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class Book:
    """Implements a Limit Order Book via array based data structure

    Two Agent Based Models are currently supported: 
    - The SFGK Zero Intelligence Model c.f 
    - The Cont-Stoikov-Talreja Model c.f

    Currently only supports placing orders of size 1
    """
    def __init__(self):
        """Default Constructor used to initialize the order book"""
        self._levels = 1000
        self._depth = 5
        self.price = np.arange(-self._levels, self._levels + 1)
        self.buy_orders, self.sell_orders = self._initialize_orders()

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

        :return: buy/sell orders on the bid/ask side
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
        return self.price[self.sell_orders > 0].min()

    def best_bid(self) -> float:
        """Computes the best bid on bid side

        :return: the best bid price is returned
        :rtype: float
        """
        return self.price[self.buy_orders > 0].max()

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
        p = self.best_ask()
        self.sell_orders[self.price == p] = self.sell_orders[self.price == p][0] - qty

    def market_sell(self, qty: int = 1):
        """Places market sell order at best bid

        :param qty: the quantity to sell
         if not specified, defaults to 1.
        :type qty: int, optional
        """
        p = self.best_bid()
        self.buy_orders[self.price == p] = self.buy_orders[self.price == p][0] - qty

    def limit_buy(self, price: int, qty: int = 1):
        """Places limit buy order at a given price level

        :param price: the price at which to place the LBO
        :type price: int
        :param qty: the quantity agent wishes to buy
         if not specified, defaults to 1
        :type qty: int, optional
        """
        self.buy_orders[self.price == price] = self.buy_orders[self.price == price][0] + qty

    def limit_sell(self, price: int, qty: int = 1):
        self.sell_size[self.price == price] = self.sell_size[self.price == price][0] + 1

    def cancel_buy(self, price=None):
        if price is not None:
            self.buy_size[self.price == price] = self.buy_size[self.price==price][0] - 1
        else:
            q = self.choose(nb)
            tmp = self.buy_size[::-1].cumsum()
            posn = len(tmp[tmp >= q])
            p = self.price[posn-1]
            self.buy_size[posn-1] = self.buy_size[posn-1]-1

    def cancel_sell(self, price=None):
        if price is not None:
            self.sell_size[self.price == price] = self.sell_size[self.price == price][0] - 1
        else:
            q = self.choose(ns)
            tmp = self.sell_size.cumsum()
            posn = len(tmp[tmp < q]) + 1
            p = self.price[posn - 1]
            self.sell_size[posn - 1] = self.sell_size[posn - 1] - 1

    # functions to find the bid/ask positions and mid position
    def bid_posn(self): 
        return len(self.buy_size[self.price <= self.best_bid()])

    def ask_posn(self): 
        return len(self.sell_size[self.price <= self.best_ask()])

    def mid_posn(self): 
        return int(np.floor((self.bid_posn() + self.ask_posn())*0.5))

    def book_shape(self, band):
        """
        returns the #of shares up to band on each side of the book around mid price
        """
        buy_qty = self.buy_size[self.mid_posn() + np.arange(-band-1, 0)]
        sell_qty = self.sell_size[self.mid_posn() + np.arange(band)]
        total_qty = np.zeros(2*band+1)
        total_qty[:band+1] = buy_qty
        total_qty[band+1:] = sell_qty
        return total_qty

    def book_plot(self, band): 
        plt.plot(np.arange(-band, band+1), self.book_shape(band))

    def choose(self, m, prob=None):
        if prob is None:
            return np.random.choice(np.arange(1, m+1),1)[0]
        return np.random.choice(np.arange(1, m+1), p=prob, size=1)[0]

    def cst_model(self, mu, lamdas, thetas, L):
        cb = self.buy_size[self.price >= (self.best_ask()-L)][:L][::-1]
        cs = self.sell_size[self.price <= (self.best_bid() + L)][-L:]
        nb = cb.sum()
        ns = cs.sum()
        cb_rates = np.dot(thetas, cb)
        cs_rates = np.dot(thetas, cs)

        cum_lam = lamdas.sum()

        cum_rate = 2*mu + 2*cum_lam + cb_rates+cs_rates
        pevent = np.array([mu, mu, cum_lam, cum_lam, cb_rates, cs_rates]) / cum_rate
        ans = np.random.choice(6, 1, p=pevent)[0]

        if ans == 0:
            self.market_buy()
        elif ans == 1:
            self.market_sell()
        elif ans == 2:
            pevent = lamdas / cum_lam
            q = self.choose(L, prob = pevent)
            p = self.best_ask() - q
            self.limit_buy(price = p)
        elif ans == 3:
            pevent = lamdas / cum_lam
            q = self.choose(L, prob = pevent)
            p = self.best_bid() + q
            self.limit_sell(price = p)
        elif ans == 4:
            pevent = (thetas * cb) / cb_rates
            q = self.choose(L,prob=pevent)
            p = self.best_ask() - q
            self.cancel_buy(price = p)
        elif ans == 5:
            pevent = (thetas * cs) / cs_rates
            q = self.choose(L,prob = pevent)
            p = self.best_bid() + q
            self.cancel_sell(price = p)

    def cst_simulate(self, mu, lamdas, thetas, L, numEvents):
        """returns the average book shape"""
        # burn in for 1000 events
        n = 1000
        for i in range(n):
            self.cst_model(mu,lamdas,thetas,L)
        avgBookShape = self.book_shape(L) / numEvents

        for i in range(1,numEvents):
            self.cst_model(mu,lamdas,thetas,L)
            avgBookShape = avgBookShape + self.book_shape(L)/ numEvents
        ans = (avgBookShape[:L][::-1]+avgBookShape[L+1:])*0.5
        ans = np.append(avgBookShape[L],ans)
        return ans

    def powerlawfit(self, emp_estimates, distance):
        obj_func = lambda k,alpha,i: k*i**(-alpha)
        import lmfit as lmfit
        model = lmfit.Model(obj_func,independent_vars=['i'],param_names=['k','alpha'])
        fit = model.fit(emp_estimates,i=np.arange(1,6),k=1.2,alpha=0.4,verbose=False)
        return fit.values['k']*distance**-fit.values['alpha']

    def prob_mid(self, n=10000, xb=1, xs=1):
        """ calculates probability of mid-price to go up"""
        def send_orders(xb, xs, mu = 0.94, lamda = 1.85,theta = 0.71):
            cum_rate = 2*mu + 2*lamda + theta*xb + theta*xs
            bid_qty_down = mu+theta*xb
            ask_qty_down = mu+theta*xs
            pevent = np.array([lamda,lamda,bid_qty_down,ask_qty_down])/cum_rate
            ans = np.random.choice(np.arange(4),size=1,p=pevent)[0]

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
            qb_old,qs_old = xb,xs
            while True:
                qb_new,qs_new = send_orders(xb=qb_old,xs=qs_old)
                if qb_new == 0:
                    break
                elif qs_new == 0:
                    count += 1
                    break
                qb_old, qs_old = qb_new, qs_new
        return count / n

    def limit_order_prob(self, n=10000, xb=5, xs=5, dpos=5):
        # assumes my order is first at the bid
        def send_orders(xb, xs, d,mu=0.94, lamda=1.85, theta=0.71):
            cum_rate = 2*mu + 2*lamda + theta*(xb-1) + theta*xs
            ask_qty_down = mu+theta*xs
            pevent = np.array([lamda,lamda,mu,theta*(xb-1), ask_qty_down])/cum_rate
            ans = np.random.choice(np.arange(5), size=1, p=pevent)[0]  # pick based on respetive probabilities

            if ans == 0:  # limit buy
                xb += 1
            elif ans == 1:  # limit sell
                xs += 1
            elif ans == 2:  # market sell
                xb -= 1
                d -= 1 if d > 0 else 0
            elif ans == 3:  # cancel buy
                r = np.random.uniform()
                if r > (xb-d) / (xb-1):
                    d -= 1
                xb -= 1
            else:  # market buy
                xs -= 1
            return xb, xs, d

        count = 0
        for i in range(n):
            qb_old,qs_old,d_old = xb,xs,dpos
            while True:
                qb_new,qs_new,d_new = send_orders(xb=qb_old, xs=qs_old, d=d_old)
                if d_new == 0:  # my order has been executed
                    count += 1
                    break
                elif qs_new == 0 and d_new > 0:  # mid price has moved
                    break
                qb_old, qs_old, d_old = qb_new, qs_new, d_new
        return count / n

    def prob_making_spread(self, n=10000, xb=5, xs=5, bid_pos=5, ask_pos=5):

        def send_orders(xb, xs, bid_pos, ask_pos, mu=0.94, lamda=1.85, theta=0.71):
            xb_rate = xb - 1
            xs_rate = xs - 1
            if bid_pos == 0:
                xb_rate = xb
            if ask_pos == 0:
                xs_rate = xs

            cum_rate = 2*mu + 2*lamda + theta*xb_rate + theta*xs_rate
            pevent = np.array([lamda,lamda,mu,mu,theta*xb_rate,theta*xs_rate])/cum_rate
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
            qb_old, qs_old, b_old, a_old = xb,xs,bid_pos,ask_pos
            while True:
                qb_new, qs_new, b_new, a_new = send_orders(qb_old, qs_old, b_old, a_old)
                if b_new == 0 and a_new == 0:
                    count += 1
                    break
                elif qb_new == 0 and a_new > 0:
                    break
                elif qs_new == 0 and b_new > 0:
                    break
                qb_old, qs_old, b_old, a_old = qb_new, qs_new, b_new, a_new
        return count / n