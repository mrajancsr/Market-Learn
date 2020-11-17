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
    security : str
        security table stores list of ticker symbols and company information
    eq_daily_price : str
        this table stores daily pricing information for each security
        """

    def __init__(self):
        """default constructor used to create schema"""
        self.exchange = 'exchange'
        self.data_vendor = 'data_venddor'
        self.security = 'security'
        self.eq_daily_price = 'eq_daily_price'
        self._db = DbReader()

    def create_table_exchange(self):
        """creates the exchange table"""
        query = \
            """Create table exchange(
                id serial,
                abbrev varchar(32) not null,
                name varchar(255) not null,
                city varchar(255) null,
                currency varchar(64) null,
                timezone_offset time null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constaint pk_exchange primary key (id)
                );
            """
        # create the above table with given constraints
        self._db.execute(query)
        return query

    def create_table_data_vendor(self):
        """creates the data_vendor table"""
        query = \
            """create table data_vendor(
                id serial,
                name varchar(64) not null,
                website_url varchar(255) null,
                support_email varchar(255) null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constraint pk_vendor primary key (id)
            );
            """
        # create the above table with given constraints
        self._db.execute(query)

    def create_table_security(self):
        """creates the symbol table"""
        query = \
            """create table security(
                id serial,
                exchange_id int not null,
                ticker varchar(32) not null,
                instrument varchar(64) not null,
                name varchar(255) null,
                sector varchar(255) null,
                currency varchar(32) null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                constraint fk_exchange_id
                  foreign key (exchange_id) references exchange (id)
                constraint pk_symbol primary key (id)
            );
            """
        self._db.execute(query)

    def create_table_eq_daily_price(self):
        """creates daily equity price information"""
        query = \
            """create table eq_daily_price(
                id serial,
                data_vendor_id int not null,
                security_id int not null,
                price_date timestamp not null,
                created_date timestamp not null,
                last_updated_date timestamp not null,
                open_price numeric(19,4) null,
                high_price numeric(19,4) null,
                low_price numeric(19,4) null,
                close_price numeric(19,4) null,
                adj_close_price numeric(19,4) null,
                volume bigint null,
                constraint fk_data_vendor_id
                  foreign key (data_vendor_id) references data_vendor (id),
                constraint fk_security_id
                  foreign key (security_id) references security (id),
                constraint pk_eq_daily_price primary key (id)
                );
            """
        self._db.execute(query)

    def create_tables(self):
        """creates tables"""
        # create the exchange table and datavendor table
        self.create_table_exchange()
        self.create_table_data_vendor()

        # create the security and daily price table
        self.create_table_security()
        self.create_table_eq_daily_price()