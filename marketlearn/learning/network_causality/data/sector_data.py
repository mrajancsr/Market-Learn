"""Class to procure data for research"""

import pandas as pd
import os


class SectorPrice:
    """
    Class used to obtain data from csv for
    Hedge funds, insurance, banks and broker/dealers sectors
    - Data obtained from online repo
    """
    def __init__(self):
        """Default Constructor
        """
        pass

    def read(self):
        """read sector prices from current directory"""
        print("current directory is: {}".format(os.getcwd()))
        files = ("bdealer",
                 "bank",
                 "hedge_fund",
                 "insurance")
        path = self._set_path() + '/'
        sectors = {}
        for k in files:
            sectors[k] = pd.read_csv(path + k + '.csv', parse_dates=['Date'])
            sectors[k]['sector'] = k
            sectors[k].index = sectors[k].pop("Date")
        return sectors

    def _set_path(self):
        """path from where we download from"""
        path = '/Users/raj/Documents/mlfinlab/mlfinlab_premium/mlfinlab/'
        path = os.path.join(path, 'network_causality', 'data')
        return path
