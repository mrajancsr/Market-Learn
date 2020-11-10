"""Implementation of interpolation methods
   of term structure of Zero Rate Curve

Author: Rajan Subramanian
Date: 11/10/2020

Notes:
- The module replicates the following paper
  http://web.math.ku.dk/~rolf/HaganWest.pdf
- Currently only supports liner spot,
  log of forward rates and cubic spline
  interpolations
Assumptions:
       - Assumes all rates are continuous compounding
"""
import numpy as np


class BoundsError(Exception):
    pass


class ZeroRateCurve:
    """Builds zero-rate curve using ZCBs with continuous discounting"""

    def __init__(self,
                 maturities: np.ndarray,
                 zcb: np.ndarray = None,
                 zero_rates: np.ndarray = None):
        """Default Constructor used to initialize zero rate curve

        :param maturities: maturity corresponding to each
         zero coupon bond in the zcb array
        :type maturities: np.ndarray
         shape = (n_samples,)
        :param zcb: zero coupon bond prices traded for various maturities
        :type rates: np.ndarray
         shape = (n_samnples,)
        """
        self.maturities = maturities
        self.zcb_prices = zcb
        self.disc_factors = zcb / 100.0
        self.zero_rates = zero_rates

    def discount_factor(self, t: float, rcont: float) -> float:
        """computes the discount factor corresponding to continuous rates

        This is the maximum amount you give up today to receive
        $1 in t years

        :param t: maturity of each zcb
        :type t: float
        :param rcont: the continuous rate
        :type rcont: float
        :return: discount factor corresponding to continuous rate
        :rtype: float
        """
        return np.exp(-rcont*t)
    
    def rates_from_discount_factors(self, t: float, discount_factor: float):
        """computes the continuous rates from discount factors

        This is the rate  you would receive if you bought the zcb today
        and held it until maturity t in years

        :param t: maturity of zcb or rate
        :type t: float in years
        :param discount_factor: zcb discount factor
        :type discount_factor: float
        :return: rate corresponding to discount factors
        :rtype: float
        """
        return (1.0 / t) * np.log(1 / discount_factor)
        
    def linear_interp(self, k: float) -> float:
        """performs linear interpolation of spot rate

        :param k: maturity of payment date of $1
        :type k: float
        :return: rate corresponding to maturity t
        :rtype: float
        """
        r = self.rates
        t = self.maturities
        tmin = t[0]
        tmax = t[-1]

        if k < tmin or k > tmax:
            raise BoundsError("Maturity out of bounds")

        # find index where t1 < t < t2
        told = len(t[t < k]) - 1
        tnew = told + 1
        terms = (r[tnew] - r[told]) / (t[tnew] - t[told])
        return terms * (k - t[told]) + r[told]


        
