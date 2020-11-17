"""Module creates the Securitities Master DataBase for Equities

Author: Rajan Subramanian
Date: 11/16/2020
"""

import pandas as pd
from marketlearn.dbhelper.dbreader import DbReader


class EquitySchema:
    """Creates Schema to hold end of day equity prices

    Parameters
    ----------
    None

    Attributes
    ----------
    exchange : str
        exchange table lists exchange we wish to obtain equity
        pricing information from.
        Currently only supports NYSE and NASDAQ
    data_vendor : str
        this table lists information about historical pricing data
        vendors.  Yahoo EOD data will be used
    symbol : str
        symvol table stores list of ticker symbols and company information
    daily_price : str
        this table stores daily pricing information for each security
        """

    def __init__(self, exchange: str, data_vendor: str, symbol: str, daily_price: str):
        """default constructor used to create schema"""
        self.exchange = None
        self.data_vendor = None
        self.symbol = None
        self.daily_price = None
        self._db = DbReader()

    def create_table_exchange(self):
        """creates the exchange table"""
        query = \
            """Create Table exchange(
                exchg_id serial,
                abbrev varchar(32) not null,
                name varchar(255) not null,
                city varchar(255) null,
                currency varchar(64) null,
                timezone_offset time null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constaint pk_exchange primary_key (exchg_id)
            """
        # create the above table with given constraints
        self._db.execute(query)
    
    def create_table_data_vendor(self):
        """creates the data_vendor table"""
        pass
